
import tensorflow as tf
import numpy as np
import random
from keras import Model

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-3
eval_iters = 200
# ------------

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

#print(text)

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# print(vocab_size)
# print(chars)

# Train and test splits
data = np.array(encode(text), dtype=np.int64)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == 'train' else val_data

    ix = np.random.randint(0, len(data) - block_size, size=batch_size)

    x = np.stack([data[i: i+block_size] for i in ix])
    y = np.stack([data[i+1: i+block_size + 1] for i in ix])
    return x, y


def estimate_loss(model: Model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = np.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class BigramLanguageModel(tf.keras.Model):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = tf.keras.layers.Embedding(vocab_size, vocab_size)


    def call(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx)  # (B, T, C)

        if targets is None:
            loss = None
        else:
            B, T = logits.shape[:2]
            logits = tf.reshape(logits, (-1, logits.shape[-1]))  # (B*T, C)
            targets = tf.reshape(targets, (-1))  # (B*T,)
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(targets, logits))

        return logits, loss



def train():
    model = BigramLanguageModel(vocab_size)

    optimizer=tf.keras.optimizers.Adam(learning_rate)

    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
