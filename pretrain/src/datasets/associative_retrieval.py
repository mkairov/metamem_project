import torch
from tqdm.auto import tqdm

NUM_SYMBOLS = 16


def generate_pairs(key_size, value_size, num_pairs, num_samples, rewrite_setting):
    keys = torch.empty((num_samples, num_pairs, key_size))

    if not rewrite_setting:
        for i in tqdm(range(num_samples)):
            key = torch.randperm(NUM_SYMBOLS ** key_size)[:num_pairs]
            for j in range(key_size):
                keys[i, :, j] = key % NUM_SYMBOLS
                key //= NUM_SYMBOLS
    else:
        keys = torch.randint(0, NUM_SYMBOLS, (num_samples, num_pairs, key_size))
    
    values = torch.randint(0, NUM_SYMBOLS, (num_samples, num_pairs, value_size))
    return keys, values


class ARDataset:
    def __init__(self, key_size, value_size, num_pairs, num_samples, mode='remember'):
        self.num_pairs = num_pairs
        rewrite_setting = mode == "rewrite"
        self.keys, self.values = generate_pairs(key_size, value_size, num_pairs, num_samples, rewrite_setting)

        if mode == 'remember':
            self.target_key_idx = torch.randint(num_pairs, (num_samples, ))
        
        else:
            self.target_key_idx = torch.empty((num_samples,), dtype=torch.long)
            for i in tqdm(range(num_samples)):
                unique_keys = self.keys[i].unique(dim=0)
                key = unique_keys[torch.randperm(len(unique_keys))[0]]

                try:
                    idx = torch.max(torch.where(torch.all(self.keys[i] == key, dim=-1))[0], dim=0).long()
                except Exception:
                    print(f"{self.keys[i]}, {key}")
                    raise 1
                assert torch.all(self.keys[i][idx] == key)
                self.target_key_idx[i] = idx
    
    def __getitem__(self, idx):
        keys, values, tgt_idx = self.keys[idx], self.values[idx], self.target_key_idx[idx]
        sample = {'keys': keys, 'values': values, 'target_key_idx': tgt_idx}
        return sample
    
    def __len__(self):
        return self.keys.shape[0]

