"""
Code implementation based on the [video tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY&list=WL&index=6)
provided by Andrej Karpathy. This implementation follows the paper: "Attention is All You Need".
"""
import os
import wget
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32  # how many independent sequences will be processed in parallel
block_size = 256  # what is the maximum context length for predictions
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embd = 384
n_head = 16
n_layer = 6
dropout = 0.2
# ----------------

torch.manual_seed(1337)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device}')

if 'input.txt' not in os.listdir():
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    wget.download(url, 'input.txt')
    print('Downloaded input.txt')

# read it in to inspect it
with open('input.txt', 'r') as f:
    text = f.read()
print("length of dataset in characters: ", len(text))

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# creating mappings from/to char to/from int
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[ch] for ch in s]  # encoder: take a string, output a list of indices
decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be training, remaining will be validation
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


# @torch.no_grad disables storing the gradients for the computation
# This function is used to estimate the current loss on the train and val sets
# during training. This metric is a better measure of progress than the loss only.
# This annotation also makes evaluation faster because instructs pytorch not to
# store intermedia results, which are not needed for this function.
@torch.no_grad()
def estimate_loss():
    out = {}
    # This disables network training features, like dropout, useful for evaluation
    # although not necessary in this simple model it is a good practice
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss
        out[split] = losses.mean()
    model.train()  # This puts the network back in training mode
    return out


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # buffer makes tril not to be a parameter of the model
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        # compute attention scores ("affinities"), equation from the paper.
        wei = q @ k.transpose(-2, -1) * C ** -0.5  # (B, T, C) @ (B, C, T) = (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B, T, head_size)
        out = wei @ v  # (B, T, T) @ (B, T, head_size) = (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # project back to the original path-way to be concatenated with skip connections
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            # Paper indicates the feed forward is 4 times the embedding dimension
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            # project back to the original path-way to be concatenated with skip connections
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)  # to avoid vanishing/exploding gradients
        self.ln2 = nn.LayerNorm(n_embd)  # to avoid vanishing/exploding gradients

    def forward(self, x):
        # The summation represents the skip connections from the architecture
        # The layer-norm is applied previous to the self-attention and feed-forward,
        # this is an optimization that helps to stabilize the training and differs from
        # the original paper. This technique is called pre-layer normalization.
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class TransformerDecoderLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # i.e. 4 heads of 8-dimensional self-attention
        # this is similar to group convolution in CNN
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # to avoid vanishing/exploding gradients
        self.lm_head = nn.Linear(n_embd, vocab_size)  # Indirection layer to got from embeddings to logits

    def forward(self, idx, targets=None):
        # B = batch_size, T = block_size, C = n_embd
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        # pos_emb positional numbers from 0 to T-1
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C), broadcasting over batch_size
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (batch_size, block_size) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens, so it doesn't overflow
            # the max size of the attention head masking
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # Becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = TransformerDecoderLanguageModel()
m = model.to(device)

print("Before training =============== ")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
print("================================")

# create a pytorch optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for iter in range(max_iters + 1):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f'step {iter}: train loss {losses["train"]:.4f}, val loss {losses["val"]:.4f}')

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
print("After training: =============== ")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
print("================================")
