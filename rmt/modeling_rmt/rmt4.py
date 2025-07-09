import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from lm_experiments_tools.utils import RMTOutput

from accelerate.logging import get_logger
logger = get_logger('')


class MemoryLayerWrapper(nn.Module):
    def __init__(self, layer, num_mem_tokens, segment_size, memory_dim, embd_std=0.02, init_inner_lr=0.1, init_stability_coef=0.02, inner_steps=10, **kwargs):
        super().__init__()
        self.layer = layer

        self.num_mem_tokens = num_mem_tokens
        self.segment_size = segment_size
        self.create_memory(memory_dim, embd_std)

        self.inner_steps = inner_steps
        # meta-learnable parameters (log-space for power scaling)
        self.log_inner_lr = nn.Parameter(torch.log(torch.tensor(init_inner_lr)))
        self.log_stability_coef = nn.Parameter(torch.log(torch.tensor(init_stability_coef)))

        self.inner_clip_value = kwargs.get('inner_clip_value', None)
        self.inner_clip_norm = kwargs.get('inner_clip_norm', None)

        # self.inner_lr = init_inner_lr
        # self.stability_coef = init_stability_coef

        self.memory_state = None

        # self.memory_mlp = nn.Sequential(
        #     nn.Linear(in_features=memory_dim, out_features=4 * memory_dim, bias=True),
        #     nn.ReLU(),
        #     nn.Linear(in_features=4 * memory_dim, out_features=memory_dim, bias=True),
        # )

        self.fc1 = torch.nn.Linear(memory_dim, 4 * memory_dim)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(4 * memory_dim, memory_dim)

        torch.nn.init.xavier_normal_(self.fc1.weight)  # Apply to the first layer
        torch.nn.init.xavier_normal_(self.fc2.weight)  # Apply to the second layer
        torch.nn.init.zeros_(self.fc1.bias)  # Set bias to zero
        torch.nn.init.zeros_(self.fc2.bias)

        self.generate_mode = False
        self.first_inputs = True
        self.lnum = kwargs.get('lnum', 0)

        self.stats = {}

    def memory_mlp(self, memory_state):
        return self.fc2(self.relu(self.fc1(memory_state)))

    def create_memory(self, memory_dim, embd_std):
        memory_weights = torch.randn((self.num_mem_tokens, memory_dim)) * embd_std
        self.register_parameter('layer_memory', torch.nn.Parameter(memory_weights, requires_grad=True))

        hidden_weights = torch.randn((self.segment_size, memory_dim)) * embd_std
        self.register_parameter('hidden_init', torch.nn.Parameter(hidden_weights, requires_grad=True))

    def set_memory(self, input_shape):
        memory = self.layer_memory.repeat(input_shape[0], 1, 1)
        return memory
    
    def set_hidden_init(self, input_shape):
        hidden_init = self.hidden_init.repeat(input_shape[0], 1, 1)
        return hidden_init
    
    def _sgd_step(self, p, g, clip_value=None, clip_norm=None):
        if clip_value is not None:
            # simple element-wise clamp
            g = torch.clamp(g, -clip_value, clip_value)

        if clip_norm is not None:
            # scale gradient if its 2-norm is too large
            # check grad for each sample separately as we do per-sample optimization
            g_norm = g.norm(dim=[1, 2], keepdim=True)                 # (B,1,1)
            scale = clip_norm / (g_norm + 1e-6)
            g = torch.where(g_norm > clip_norm, g * scale, g)

        inner_lr = torch.exp(self.log_inner_lr)
        return p - inner_lr * g
    
    @torch.enable_grad()
    def inner_loop(self, hidden_ref, attention_mask, *args, **kwargs):
        # logger.info(f"{hidden_ref.shape}, {attention_mask.shape}")
        # logger.info(args)
        # logger.info(kwargs)
        # mem_live = self.memory_state.clone().requires_grad_(True)
        self.memory_state.requires_grad_(True)

        # freeze transformer layer
        # for p in self.layer.parameters():
        #     p.requires_grad = False

        # get exponents of log-values
        stability_coef = torch.exp(self.log_stability_coef)
        hidden_init = self.set_hidden_init(hidden_ref.shape).requires_grad_(True)
        memory_ref = self.memory_state.clone().detach().requires_grad_(False)
        
        device = hidden_ref.device
        self.stats['inner_grad_norm_mean'] = torch.tensor(0.0, device=device)
        self.stats['recon_loss'] = torch.tensor(0.0, device=device)
        self.stats['stability_loss'] = torch.tensor(0.0, device=device)
        for _ in range(self.inner_steps):
            inputs = torch.cat([self.memory_state, hidden_init], dim=1) # concat along seq_len
            # logger.info(f"{inputs.shape}, {kwargs.get('attention_mask').shape}")
            out = self.layer(hidden_states=inputs, attention_mask=attention_mask, *args, **kwargs)
            out_recon = out[0][:, self.num_mem_tokens:, :]
            # logger.info(f"layer {self.lnum}: {inputs.shape}, {out_recon.shape}, {out_ref.shape}, {input_ref.shape}")

            # prediction loss + stability loss
            loss_recon = F.mse_loss(out_recon, hidden_ref)
            loss_stab = F.mse_loss(self.memory_state, memory_ref)
            total_loss = loss_recon + stability_coef * loss_stab

            # manual gradient update on mem_live
            g = torch.autograd.grad(total_loss, self.memory_state, create_graph=True)[0]
            g_norm = g.reshape(hidden_ref.shape[0], -1).norm(dim=1).detach()
            self.stats['inner_grad_norm_mean'] += g_norm.mean()
            self.stats['recon_loss'] = loss_recon
            self.stats['stability_loss'] = loss_stab
            # mem_live = mem_live - inner_lr * grads
            self.memory_state = self._sgd_step(self.memory_state, g, clip_value=self.inner_clip_value, clip_norm=self.inner_clip_norm)
        
        self.stats['inner_mem_norm_mean'] = self.memory_state.norm(dim=[1, 2]).detach().mean()
        self.stats['inner_grad_norm_mean'] /= self.inner_steps
        self.stats['inner_final_loss'] = total_loss
    
        return self.memory_state

    def forward(self, hidden_states, attention_mask, *args, **kwargs):
        if self.memory_state is None:
            self.memory_state = self.set_memory(hidden_states.shape).requires_grad_(True)
        hidden_states = hidden_states[:, self.num_mem_tokens:]
        
        is_last_segment = kwargs.get('is_last_segment', False) or self.generate_mode
        if is_last_segment:
            # unfreeze layer: we want to learn to retrieve from memory
            for p in self.layer.parameters():
                p.requires_grad = True

            # when using cache, do not prepend memory
            if self.generate_mode and self.first_inputs or not self.generate_mode:
                # only one memory prepended

                # if self.lnum < 3:
                # logger.info(f"layer {self.lnum}: {hidden_states.shape}, {self.memory_state.shape}")
                self.memory_state = self.memory_mlp(self.memory_state)
                hidden_states = torch.cat([self.memory_state, hidden_states], dim=1)
                self.first_inputs = False
            
            out = self.layer(hidden_states=hidden_states, attention_mask=attention_mask, *args, **kwargs)
            return out

        # clone memory, prepend to hidden_states
        # freeze transformer layer, do ttt loop
        # get updated memory, prepend to hs, forward pass

        # get reference memory
        # with torch.no_grad():
        input_ref = torch.cat([self.memory_state, hidden_states], dim=1)  # (B, M+L, D)
        out = self.layer(hidden_states=input_ref, attention_mask=attention_mask, *args, **kwargs)
        hidden_ref = out[0][:, self.num_mem_tokens:]

        # with torch.enable_grad():
            # hidden_states = hidden_states[:, self.num_mem_tokens:]
            # append to hidden states, pass through layer – we'll approximate
            # logger.info(f"gradnorm: {grads.norm()}, loss: {total_loss.item()}")
        
        self.memory_state = self.inner_loop(hidden_ref, attention_mask=attention_mask, *args, **kwargs)
        # mem_live = mem_live.detach().requires_grad_(False)
        # self.memory_state = self.memory_state + mem_live - mem_ref

        # input_final = torch.cat([self.memory_state, hidden_states], dim=1)
        # out_final = self.layer(hidden_states=input_final, attention_mask=attention_mask, **kwargs)
        return out

    def reset_memory(self):
        self.memory_state = None


class MemoryCell(nn.Module):
    def __init__(self, base_model, num_mem_tokens, layers_attr: str = 'transformer.h', **kwargs):
        super().__init__()
        self.model = base_model
        self.num_mem_tokens = num_mem_tokens

        embd_std = self.model.get_input_embeddings().weight.data.std().cpu().item()

        self.layers = self.model
        self.layers_attrs = layers_attr.split('.')
        for i, attr in enumerate(self.layers_attrs):
            self.layers = getattr(self.layers, attr)

        for i in range(len(self.layers)):
            self.layers[i] = MemoryLayerWrapper(
                layer=self.layers[i],
                num_mem_tokens=num_mem_tokens,
                memory_dim=self.model.config.hidden_size,
                embd_std=embd_std,
                lnum=i,
                **kwargs
            )

    def forward(self, input_ids, **kwargs):
        seg_kwargs = self.process_input(input_ids, **kwargs)
        if kwargs.get('is_last_segment', False):
            for n, p in self.model.named_parameters():
                p.requires_grad = True
        else:
            for n, p in self.model.named_parameters():
                if 'layer_memory' not in n.lower() and 'hidden_init' not in n.lower():
                    p.requires_grad = False

        out = self.model(**seg_kwargs, use_cache=False)
        out = self.process_output(out, **kwargs)
        return out
    
    def reset_memory(self):
        for layer in self.layers:
            layer.reset_memory()
    
    def generate_mode(self, mode):
        for layer in self.layers:
            layer.generate_mode = mode
            layer.first_inputs = True
    
    def generate(self, input_ids, attention_mask=None, **generate_kwargs):
        self.generate_mode(True)
        seg_kwargs = self.process_input(input_ids, attention_mask=attention_mask)
        out = self.model.generate(inputs_embeds=seg_kwargs['inputs_embeds'], attention_mask=seg_kwargs['attention_mask'], **generate_kwargs, use_cache=False)
        self.generate_mode(False)
        return out

    def process_input(self, input_ids, **kwargs):
        seg_kwargs = dict(**kwargs)

        inputs_embeds = kwargs.get('inputs_embeds', None)
        device = next(self.parameters()).device
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids.to(device))
        
        blank_tokens = torch.zeros((inputs_embeds.shape[0], self.num_mem_tokens, inputs_embeds.shape[2]), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        inputs_embeds = torch.cat([blank_tokens, inputs_embeds], dim=1)

        seg_kwargs['input_ids'] = None
        seg_kwargs['inputs_embeds'] = inputs_embeds
        # logger.info(f"input embeds {seg_kwargs['inputs_embeds'].shape}")
        if kwargs.get('attention_mask') is not None:
            seg_kwargs['attention_mask'] = self.pad_attention_mask(kwargs['attention_mask'], inputs_embeds.shape).to(device)
            # seg_kwargs['attention_mask'] = kwargs.get('attention_mask')
        seg_kwargs['output_hidden_states'] = True
        return seg_kwargs
    
    def process_output(self, model_outputs, **kwargs):
        out = RMTOutput()
        out['logits'] = model_outputs.logits[:, self.num_mem_tokens:]
        # logger.info(f"logits {out['logits'].shape}")
        
        if kwargs.get('output_hidden_states'):
            out['hidden_states'] = [lh[:, self.num_mem_tokens:] for lh in model_outputs.hidden_states]
        if kwargs.get('output_attentions'):
            out['attentions'] = model_outputs['attentions']
        
        inner_stats = self.collect_inner_stats()
        for k, v in inner_stats.items():
            out[k] = v

        return out 
    
    def pad_attention_mask(self, attention_mask, shape):
        shape = list(shape)[:2]
        mask = torch.ones(*shape, dtype=torch.int64).to(attention_mask.device)
        mask[:, -attention_mask.shape[1]:] = attention_mask
        return mask
        
    def manage_gradients(self, *args, **kwargs):
        for layer in self.layers:
            layer.memory_state = self._manage_gradients(layer.memory_state, *args, **kwargs)
        
    def _manage_gradients(self, memory_state, seg_num, k2, max_n_segments):
        if seg_num == 0 \
            or k2 in {-1, None} \
            or seg_num + k2 > max_n_segments:
                return memory_state
        
        memory_state = memory_state.detach()
        return memory_state
    
    def collect_inner_stats(self):
        stats = []
        for layer in self.layers:
            stats.append(layer.stats)
            layer.stats = {}
        
        stats_mean = {}
        keys = stats[0].keys()
        for k in keys:
            stats_mean[k] = np.mean([s[k].cpu().item() for s in stats])
        return stats_mean


class RecurrentWrapper(nn.Module):
    def __init__(self, memory_cell, **rmt_kwargs):
        super().__init__()
        self.memory_cell = memory_cell
        self.rmt_config = rmt_kwargs

    def forward(self, input_ids, labels=None, labels_mask=None, inputs_embeds=None, attention_mask=None, output_attentions=None, output_hidden_states=None):
        segmented = self.segment(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask)

        cell_outputs = []
        self.memory_cell.reset_memory()
        for seg_num, segment in enumerate(segmented):
            is_last_segment = seg_num == len(segmented) - 1
            cell_out = self.memory_cell(**segment, output_attentions=output_attentions, output_hidden_states=True, is_last_segment=is_last_segment)
            cell_outputs.append(cell_out)
            self.memory_cell.manage_gradients(seg_num, self.rmt_config.get('k2'), self.rmt_config.get('max_n_segments'))

        out = self.process_outputs(cell_outputs, labels=labels, 
                                   labels_mask=labels_mask,
                                   output_attentions=output_attentions, 
                                   output_hidden_states=output_hidden_states)
        return out
    
    def generate(self, input_ids, attention_mask=None, **generate_kwargs):
        segmented = self.segment(input_ids=input_ids, attention_mask=attention_mask)

        self.memory_cell.reset_memory()
        for segment in segmented[:-1]:
            self.memory_cell(**segment)

        final_segment = segmented[-1]
        out = self.memory_cell.generate(**final_segment, **generate_kwargs)

        return out

    def segment(self, **kwargs):
        segments = []
        for k, tensor in kwargs.items():
            if tensor is not None:
                k_segments = self.split_tensor(tensor)
                for s, k_seg in enumerate(k_segments):
                    if s < len(segments):
                        segments[s][k] = k_seg
                    else:
                        segments.append({k: k_seg})

        return segments
    
    def split_tensor(self, tensor):
        align = self.rmt_config.get('segment_alignment')
        segment_size = self.rmt_config.get('segment_size')
        if align in {'left', None}:
            split_inds = list(range(0, tensor.shape[1], segment_size)) + [tensor.shape[1]]
            segments = [tensor[:, start:end] for (start, end) in zip(split_inds, split_inds[1:])]
        elif align in {'right', None}:
            split_inds = (list(range(tensor.shape[1], 0, -segment_size)) + [0])[::-1]
            segments = [tensor[:, start:end] for (start, end) in zip(split_inds, split_inds[1:])]
        elif align == 'center':
            n_seg = math.ceil(tensor.shape[1] / segment_size)
            segments = torch.chunk(tensor, n_seg, dim=1)
        else:
            raise NotImplementedError
        return segments

    def process_outputs(self, cell_outputs, **kwargs):
        out = RMTOutput()
        full_logits = torch.cat([o.logits for o in cell_outputs], dim=1)
        full_hidden_states = tuple([torch.cat(layer_hs, dim=1) for layer_hs in zip(*[o.hidden_states for o in cell_outputs])])

        # logger.info(kwargs.get('labels_mask').shape)
        # logger.info(kwargs.get('labels').shape)

        labels = kwargs.get('labels')
        # logger.info(f"labels {labels.shape}")
        # logger.info(f"mask {kwargs.get('labels_mask').shape}")
        if labels is not None:
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = full_logits[..., :-1, :].contiguous()
            flat_labels = shift_labels.view(-1)
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            
            loss_fct = nn.CrossEntropyLoss()
            labels_mask = kwargs.get('labels_mask')

            # logger.info(f"{shift_labels.shape}, {shift_logits.shape}, {flat_labels.shape}, {flat_logits.shape}")

            if labels_mask is not None:
                shift_mask = labels_mask[..., :-1].contiguous()

                flat_labels = flat_labels[shift_mask.view(-1)]
                flat_logits = flat_logits[shift_mask.view(-1)]
     
            out['loss'] = loss_fct(flat_logits, flat_labels)
            if out['loss'] is None:
                raise ValueError
        else:
            out['loss'] = 0

        out['logits'] = full_logits
        segment_keys = ['loss', 'logits', 'inner']
        if kwargs.get('output_attentions'):
            segment_keys.append('attentions')
        if kwargs.get('output_hidden_states'):
            segment_keys.append('hidden_states')
            out['hidden_states'] = full_hidden_states

        for seg_num, o in enumerate(cell_outputs):
            for key, value in o.items():
                if any([sk in key for sk in segment_keys]):
                    out[f'{key}_{seg_num}'] = value

        out['inner_stability_coef'] = np.mean([torch.exp(l.log_stability_coef).item() for l in self.memory_cell.layers])
        out['inner_lr'] = np.mean([torch.exp(l.log_inner_lr).item() for l in self.memory_cell.layers])

        return out
