import torch
from torch import nn
import torch.nn.functional as F

act_map = {
    'relu': nn.ReLU(),
    'gelu': nn.GELU(),
    'tanh': nn.Tanh(),
}

class MLP(nn.Module):

    def __init__(self, n_embd, n_proj, act):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, n_proj),
            act_map[act],
            nn.Linear(n_proj, n_embd),
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):

    def __init__(self, n_embd, block_size, n_head, bias):
        super().__init__()

        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        B, T, C = x.shape # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * k.shape[-1]**-0.5
        att = att.masked_fill(self.tril[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        out = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        out = out.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        return self.c_proj(out)

class Model(nn.Module):

    def __init__(self, vocab_size, block_size, n_embd, n_head, bias, n_proj, act): # TODO: take a object containing the custom architecture structure and parameters
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.architecture = nn.Sequential(
            Attention(n_embd, block_size, n_head, bias),
            MLP(n_embd, n_proj, act),
            ) # TODO: implement code to build the custom architecture
    
    def forward(self, x):
        x = self.token_embedding(x) + self.position_embedding(torch.arange(x.shape[-1]))
        x = self.architecture(x)
        return self.lm_head(x)


if __name__ == "__main__":

    with open('nn/input.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

    data = torch.tensor(encode(text), dtype=torch.long)

    block_size = 64
    tokenized_input = data[:block_size].unsqueeze(0)
    print("Tokenized Input: \n", tokenized_input)
    print("Tokenized Input Shape: \n", tokenized_input.shape)

    n_embd = 16
    n_head = 4
    bias = False
    n_proj = 32
    act = 'gelu'
    model = Model(vocab_size, block_size, n_embd, n_head, bias, n_proj, act)
    model_output = model(tokenized_input)
    print("Model Output: \n", model_output)
    print("Model Output Shape: \n", model_output.shape)