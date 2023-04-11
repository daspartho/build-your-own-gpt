import torch
from torch import nn

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

class Model(nn.Module):

    def __init__(self, vocab_size, block_size, n_embd, n_proj, act): # TODO: take a object containing the custom architecture structure and parameters
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.architecture = nn.Sequential(MLP(n_embd, n_proj, act)) # TODO: implement code to build the custom architecture
    
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
    n_proj = 32
    act = 'gelu'
    model = Model(vocab_size, block_size, n_embd, n_proj, act)
    model_output = model(tokenized_input)
    print("Model Output: \n", model_output)
    print("Model Output Shape: \n", model_output.shape)