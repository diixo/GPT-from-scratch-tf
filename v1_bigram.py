
import tensorflow as tf
import numpy as np
import random
from keras import Model

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
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
    for split in ['train', 'val']:
        losses = np.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            # Передаем training=False для режима оценки (без dropout, batch norm и т.п.)
            logits, loss = model(X, Y)
            losses[k] = loss.numpy()  # Получаем числовое значение из тензора
        out[split] = losses.mean()
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


    def generate(self, idx, max_new_tokens):
        # idx — это (B, T) тензор с индексами текущего контекста
        for _ in range(max_new_tokens):
            # Получаем предсказания: logits имеет форму (B, T, C)
            logits, _ = self(idx)
            # Берем логиты только последнего временного шага: (B, C)
            logits = logits[:, -1, :]
            # Применяем softmax для получения вероятностей
            probs = tf.nn.softmax(logits, axis=-1)  # (B, C)
            # Сэмплируем следующий токен из распределения вероятностей.
            # tf.random.categorical принимает логиты, поэтому применяем log(probs)
            idx_next = tf.random.categorical(tf.math.log(probs), num_samples=1)  # (B, 1)
            # Добавляем сэмплированный токен к последовательности
            idx = tf.concat([idx, idx_next], axis=1)  # (B, T+1)
        return idx


def train(model):

    optimizer = tf.keras.optimizers.Adam(learning_rate)
    
    for iter in range(max_iters):

        if iter % eval_interval == 0:
            losses = estimate_loss(model)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        xb, yb = get_batch('train')

        # Обучаем модель
        with tf.GradientTape() as tape:
            logits, loss = model(xb, yb)
        
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))


model = BigramLanguageModel(vocab_size)

train(model)

context = np.zeros((1, 1), dtype=np.int64)
idxs = model.generate(context, max_new_tokens=500)
print(decode(idxs[0].numpy()))

