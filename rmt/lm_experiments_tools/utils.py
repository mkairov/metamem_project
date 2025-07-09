import functools
import importlib
import inspect
import json
import logging
import os
import platform
import subprocess
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List

import torch
import transformers

import string
from tokenizers import Tokenizer, Regex
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from transformers import PreTrainedTokenizerFast

import lm_experiments_tools.optimizers

# import schedulefree
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from typing import Optional, Tuple
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer


def get_cls_by_name(name: str) -> type:
    """Get class by its name and module path.

    Args:
        name (str): e.g., transfomers:T5ForConditionalGeneration, modeling_t5:my_class

    Returns:
        type: found class for `name`
    """
    module_name, cls_name = name.split(':')
    return getattr(importlib.import_module(module_name), cls_name)


def get_git_hash_commit() -> str:
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except subprocess.CalledProcessError:
        # no git installed or we are not in repository
        commit = ''
    return commit


def get_git_diff() -> str:
    try:
        diff = subprocess.check_output(['git', 'diff', 'HEAD', '--binary']).decode('utf8')
    except subprocess.CalledProcessError:
        # no git installed or we are not in repository
        diff = ''
    return diff


def get_fn_param_names(fn) -> List[str]:
    """get function parameters names except *args, **kwargs

    Args:
        fn: function or method

    Returns:
        List[str]: list of function parameters names
    """
    params = []
    for p in inspect.signature(fn).parameters.values():
        if p.kind not in [inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD]:
            params += [p.name]
    return params


def get_optimizer(name: str):
    if ':' in name:
        return get_cls_by_name(name)
    if hasattr(lm_experiments_tools.optimizers, name):
        return getattr(lm_experiments_tools.optimizers, name)
    if hasattr(torch.optim, name):
        return getattr(torch.optim, name)
    if hasattr(transformers.optimization, name):
        return getattr(transformers.optimization, name)
    # if hasattr(schedulefree, name):
    #     return getattr(schedulefree, name)
    try:
        apex_opt = importlib.import_module('apex.optimizers')
        return getattr(apex_opt, name)
    except (ImportError, AttributeError):
        pass
    return None


def collect_run_configuration(args, env_vars=['CUDA_VISIBLE_DEVICES']):
    args_dict = dict(vars(args))
    args_dict['ENV'] = {}
    for env_var in env_vars:
        args_dict['ENV'][env_var] = os.environ.get(env_var, '')
    # hvd
    try:
        import horovod.torch as hvd
        args_dict['HVD_INIT'] = hvd.is_initialized()
        if hvd.is_initialized():
            args_dict['HVD_SIZE'] = hvd.size()
    except ImportError:
        pass
    # accelerate
    # todo: collect full accelerate config
    try:
        import accelerate
        args_dict['accelerate'] = {}
        args_dict['accelerate']['initialized'] = accelerate.PartialState().initialized
        if accelerate.PartialState().initialized:
            args_dict['accelerate']['num_processes'] = accelerate.PartialState().num_processes
            args_dict['accelerate']['backend'] = accelerate.PartialState().backend
            args_dict['accelerate']['distributed_type'] = accelerate.PartialState().distributed_type
    except ImportError:
        pass

    args_dict['MACHINE'] = platform.node()
    args_dict['COMMIT'] = get_git_hash_commit()
    return args_dict


def get_distributed_rank() -> int:
    try:
        import accelerate
        if accelerate.PartialState().initialized:
            return accelerate.PartialState().process_index
    except ImportError:
        pass

    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()

    try:
        import horovod.torch as hvd
        if hvd.is_initialized():
            return hvd.rank()
    except ImportError:
        pass

    return 0


def rank_0(fn):
    @functools.wraps(fn)
    def rank_0_wrapper(*args, **kwargs):
        if get_distributed_rank() == 0:
            return fn(*args, **kwargs)
        return None
    return rank_0_wrapper


def wait_for_everyone():
    try:
        import accelerate
        if accelerate.PartialState().initialized:
            accelerate.PartialState().wait_for_everyone()
    except ImportError:
        pass

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    try:
        import horovod.torch as hvd
        if hvd.is_initialized():
            hvd.barrier()
    except ImportError:
        pass


def prepare_run(args, logger=None, logger_fmt: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                add_file_logging=True):
    """creates experiment directory, saves configuration and git diff, setups logging

    Args:
        args: arguments parsed by argparser, model_path is a required field in args
        logger: python logger object
        logger_fmt (str): string with logging format
        add_file_logging (bool): whether to write logs into files or not
    """

    # create model path and save configuration
    rank = get_distributed_rank()
    if rank == 0 and args.model_path is not None:
        model_path = Path(args.model_path)
        if not model_path.exists():
            Path(model_path).mkdir(parents=True)
        args_dict = collect_run_configuration(args)
        # todo: if model path exists and there is config file, write new config file aside
        json.dump(args_dict, open(model_path / 'config.json', 'w'), indent=4)
        open(model_path / 'git.diff', 'w').write(get_git_diff())

    # configure logging to a file
    if args.model_path is not None and logger is not None and add_file_logging:
        # sync workers to make sure that model_path is already created by worker 0
        wait_for_everyone()
        # RotatingFileHandler will keep logs only of a limited size to not overflow available disk space.
        # Each gpu worker has its own logfile.
        # todo: make logging customizable? reconsider file size limit?
        fh = RotatingFileHandler(Path(args.model_path) / f"{time.strftime('%Y.%m.%d_%H:%M:%S')}_rank_{rank}.log",
                                 mode='w', maxBytes=100*1024*1024, backupCount=2)
        logger_with_fh = logger
        if isinstance(logger, logging.LoggerAdapter):
            logger_with_fh = logger.logger
        fh.setLevel(logger_with_fh.level)
        fh.setFormatter(logging.Formatter(logger_fmt))
        logger_with_fh.addHandler(fh)

    if rank == 0 and args.model_path is None and logger is not None:
        logger.warning('model_path is not set: config, logs and checkpoints will not be saved.')


class ObjectView(dict):
    def __init__(self, *args, **kwargs):
        super(ObjectView, self).__init__(**kwargs)
        for arg in args:
            if not arg:
                continue
            elif isinstance(arg, dict):
                for key, val in arg.items():
                    self[key] = val
            else:
                raise TypeError()
        for key, val in kwargs.items():
            self[key] = val

    def __setattr__(self, key, value):
        if not hasattr(ObjectView, key):
            self[key] = value
        else:
            raise

    def __setitem__(self, name, value):
        value = ObjectView(value) if isinstance(value, dict) else value
        super(ObjectView, self).__setitem__(name, value)

    def __getattr__(self, item):
        return self.__getitem__(item)

    def __getitem__(self, name):
        if name not in self:
            self[name] = {}
        return super(ObjectView, self).__getitem__(name)

    def __delattr__(self, name):
        del self[name]


@dataclass
class RMTOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    memory_states: Optional[list[torch.FloatTensor, ...]] = None


class DummyAttentionOutput:
    def __init__(self, data):
        self.data = data
    
    def split(self, *args, **kwargs):
        return self.data


def babi_collate_fn(batch):
    # tokenizer = get_tokenizer()
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    id_pad_value = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    gen_token = tokenizer.encode('GEN')[0]
    eos_token = tokenizer.eos_token_id

    targets = [torch.tensor(b['target_tokens']) for b in batch]
    
    input_ids = [torch.tensor(b['input_tokens'] + b['question_tokens'] + [gen_token] + b['target_tokens'] + [eos_token]) for b in batch]
    gen_inputs = [torch.tensor(b['input_tokens'] + b['question_tokens'] + [gen_token]) for b in batch]

    attention_mask = [torch.ones_like(b, dtype=int) for b in input_ids]
    labels_mask = [torch.zeros_like(b, dtype=bool) for b in input_ids]
    for m, t in zip(labels_mask, targets):
        m[-len(t) - 2:] = True

    input_ids = pad_sequence(input_ids, padding_value=id_pad_value, batch_first=True)
    gen_inputs = pad_sequence(gen_inputs, padding_value=id_pad_value, batch_first=True)
    attention_mask = pad_sequence(attention_mask, padding_value=0, batch_first=True)
    labels_mask = pad_sequence(labels_mask, padding_value=0, batch_first=True)

    collated = {}
    collated['input_ids'] = collated['labels'] = input_ids
    collated['input_ids_generate'] = gen_inputs
    collated['labels_mask'] = labels_mask
    collated['attention_mask'] = attention_mask.bool()
    collated['attention_mask_generate'] = (gen_inputs != id_pad_value).bool()
    collated['target_text'] = [b['answer'] for b in batch]
    return collated


def ar_collate_fn(batch, valid=False, vary_n_segments=False, rewrite_setting=False,
                  sep_token=None, gen_token=None, eos_token=None, value_size=None):
        keys = [b['keys'] for b in batch]
        values = [b['values'] for b in batch]
        
        if not vary_n_segments:
            tgt_inds = [b['target_key_ind'].item() for b in batch]
            n = len(keys[0])
        else:
            n = torch.randint(1, len(keys[0])+1, size=())
            keys = [x[-n:] for x in keys]
            values = [x[-n:] for x in values]
            if not rewrite_setting:
                tgt_inds = [torch.randint(0, n, size=()).item() for _ in range(len(keys))]
            else:
                tgt_inds = []
                for i in range(len(keys)):
                    unique_keys = keys[i].unique(dim=0)
                    key = unique_keys[torch.randperm(len(unique_keys))[0]]
                    try:
                        idx = torch.max(torch.where(torch.all(keys[i] == key, dim=-1))[0], dim=0)[0].long()
                    except Exception:
                        print(f"{keys[i]}, {key}")
                        raise 1
                    assert torch.all(keys[i][idx] == key)
                    tgt_inds.append(idx)

        bs = len(keys)
        sep_tokens = torch.ones(bs, 1) * sep_token
        eos_tokens = torch.ones(bs, 1) * eos_token
        gen_tokens = torch.ones(bs, 1) * gen_token
        sample = []

        for i in range(n):
            sample.append(torch.stack([k[i] for k in keys]))
            sample.append(sep_tokens)
            sample.append(torch.stack([v[i] for v in values]))
            sample.append(eos_tokens)

        target_keys = torch.stack([k[i] for i, k in zip(tgt_inds, keys)])
        target_values = torch.stack([k[i] for i, k in zip(tgt_inds, values)])

        sample.append(target_keys)
        sample.append(gen_tokens)

        input_ids_generate = torch.cat(sample, dim=1)

        sample.append(target_values)
        sample.append(eos_tokens)
        input_ids = torch.cat(sample, dim=1)

        labels_mask = torch.zeros_like(input_ids).bool()
        labels_mask[:, -value_size - 2:] = True

        collated = {'input_ids': input_ids.long(), 
                    'input_ids_generate': input_ids_generate.long(), 
                    'attention_mask': torch.ones_like(input_ids).bool(),
                    'attention_mask_generate': torch.ones_like(input_ids_generate).bool(),
                    'labels': input_ids.long(), 
                    'labels_mask': labels_mask, 
                    }
        return collated


def create_noisy_ar_tokenizer():
    ALPHABET = string.ascii_letters + string.digits
    # Create character tokenizer
    chars = ALPHABET + '!?:|'
    special = {'[PAD]': 0, '[BOS]': 1, '[EOS]': 2, '[UNK]': 3, '|': 4}
    vocab = {ch: i + len(special) for i, ch in enumerate(chars)}
    vocab.update(special)

    tokenizer = Tokenizer(WordLevel(vocab, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Split(Regex(r'.'), behavior="isolated", invert=True)

    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token='|',
        eos_token='|',
        bos_token='[BOS]',
        unk_token='[UNK]'
    )


def noisy_ar_collate_fn(batch, tokenizer, max_seg_len=64, num_seg=4, query_len=2, target_len=2):
    contexts = []
    gen_inputs = []
    for item in batch:
        c = item['context'][:-1]
        segments = c.split('|')
        padded_context = [seg.ljust(max_seg_len, '|')[:max_seg_len] for seg in segments]
        contexts.append(''.join(padded_context) + item['query'] + item['target'])
        gen_inputs.append(''.join(padded_context) + item['query'])
    
    input_ids = tokenizer(contexts, return_tensors="pt", add_special_tokens=False, padding=False).input_ids
    gen_ids = tokenizer(gen_inputs, return_tensors="pt", add_special_tokens=False, padding=False).input_ids
    labels_mask = torch.zeros_like(input_ids)
    target_start_pos = max_seg_len * num_seg + query_len + 3  # account for ?| before query and : after
    labels_mask[:, target_start_pos:target_start_pos + target_len] = 1  # 

    collated = {}
    collated['input_ids'] = collated['labels'] = input_ids
    collated['input_ids_generate'] = gen_ids
    collated['labels_mask'] = labels_mask.bool()
    collated['attention_mask'] = (input_ids != tokenizer.convert_tokens_to_ids("|"))
    collated['attention_mask_generate'] = (gen_ids != tokenizer.convert_tokens_to_ids("|"))
    collated['target_text'] = [b['target'][:-2] for b in batch]  # account for !|
    return collated
