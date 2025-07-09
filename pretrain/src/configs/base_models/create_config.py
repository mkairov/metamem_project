import os
import json
import argparse

save_folder = "/home/user36/metamem/src/configs/base_models/exp/"
default_config = {
  "architectures": [],
  "model_type": "gpt_neox",
  # "model_type": "gpt2",
  "vocab_size": 128,
  "hidden_size": 128, 
  "num_hidden_layers": 1, 
  "num_attention_heads": 1, 
  "intermediate_size": 128, 
  "max_position_embeddings": 2048,
  "bos_token_id": 101,
  "eos_token_id": 102,
  "hidden_act": "gelu",
  "rotary_pct": 0.25,
  "rotary_emb_base": 10000,
  "attention_dropout": 0.0,
  "hidden_dropout": 0.0,
  "classifier_dropout": 0.1,
  "initializer_range": 0.02,
  "layer_norm_eps": 1e-5,
  "use_cache": True,
  "tie_word_embeddings": False,
  "use_parallel_residual": True
  }

parser = argparse.ArgumentParser()
parser.add_argument("--hidden_size", default=128)
parser.add_argument("--num_hidden_layers", default=1)
parser.add_argument("--num_attention_heads", default=1)
parser.add_argument("--arch")
args = parser.parse_args()

arch_classes = {
    "gpt_neox": "GPTNeoXForCausalLM",
    "gpt2": "GPT2LMHeadModel",
}

config = dict(**default_config)
config['hidden_size'] = int(args.hidden_size)
config['num_hidden_layers'] = int(args.num_hidden_layers)
config['num_attention_heads'] = int(args.num_attention_heads)
config['model_type'] = args.arch
config['architectures'] = [arch_classes[args.arch]]

config_name = f"{args.arch}_tiny_{args.num_hidden_layers}l{args.num_attention_heads}hd{args.hidden_size}"
print(f'Saving config {config_name}')
save_path = os.path.join(save_folder, f'{config_name}.json')
with open(save_path, 'w') as f:
    json.dump(config, f)