from torch import nn


class MemoryMLP(nn.Module):
    def __init__(self, memory_token_dim=768, memory_token_length=8):
        super(MemoryMLP, self).__init__()
        self.memory_token_dim = memory_token_dim
        self.memory_token_length = memory_token_length
        self.in_size = memory_token_dim * memory_token_length
        self.ln1 = nn.Linear(in_features=self.in_size, out_features=self.in_size * 2)
        self.ln2 = nn.Linear(in_features=self.in_size * 2, out_features=self.in_size)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)  # [1, memory_token_length * memory_token_dim]
        
        batch_size = x.size(0)
        out = self.ln2(self.relu(self.ln1(x))).reshape(batch_size, self.memory_token_length, self.memory_token_dim)
        return out
