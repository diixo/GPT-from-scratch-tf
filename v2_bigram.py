
import tensorflow as tf
import numpy as np

# Исходный текст (пример)
text = "hello world"
vocab = sorted(set(text))  # Уникальные символы
vocab_size = len(vocab)

# Маппинг символов в индексы
lookup = tf.keras.layers.StringLookup(vocabulary=vocab, output_mode='int')
inverse_lookup = tf.keras.layers.StringLookup(vocabulary=vocab, invert=True)

# Гиперпараметры
batch_size = 32  # Количество независимых потоков данных
block_size = 8   # Длина блока
n_embd = 32      # Размерность эмбеддингов
n_head = 4       # Количество голов в multi-head attention
n_layer = 4      # Число трансформерных слоев
learning_rate = 1e-3

class BigramLanguageModel(tf.keras.Model):
    def __init__(self, vocab_size, n_embd):
        super().__init__()
        self.token_embedding = tf.keras.layers.Embedding(vocab_size, n_embd)
        self.position_embedding = tf.keras.layers.Embedding(block_size, n_embd)
        self.ln = tf.keras.layers.LayerNormalization()
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs):
        batch_size, seq_len = tf.shape(inputs)[0], tf.shape(inputs)[1]
        positions = tf.range(seq_len)[tf.newaxis, :]
        
        x = self.token_embedding(inputs) + self.position_embedding(positions)
        x = self.ln(x)
        logits = self.dense(x)
        return logits

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits = self(idx)[:, -1, :]
            probs = tf.nn.softmax(logits, axis=-1)
            next_token = tf.random.categorical(tf.math.log(probs), num_samples=1)
            idx = tf.concat([idx, next_token], axis=1)
        return idx

# Создаем модель
model = BigramLanguageModel(vocab_size, n_embd)

# Компиляция модели
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)

# Пример входных данных (преобразование текста в индексы)
encoded_text = lookup(tf.strings.unicode_split(text, 'UTF-8'))
x_test = tf.convert_to_tensor(np.tile(encoded_text[:block_size], (batch_size, 1)))

# Проверяем выходные логиты
y_pred = model(x_test)
print(y_pred.shape)  # (batch_size, block_size, vocab_size)

# Проверяем генерацию текста
start_token = tf.convert_to_tensor([[lookup('h')]])
generated_sequence = model.generate(start_token, max_new_tokens=20)

decoded_sequence = inverse_lookup(generated_sequence[0])
print("".join(decoded_sequence.numpy().astype(str)))
