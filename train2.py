import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as utils
import numpy as np
import math
import pyarrow.parquet as pq
import os
from glob import glob
import pickle
import time

### hyper params
# model
ctx_len = 128
n_emb = 128
dropout = 0.1
head_size = 128
n_heads = 4
n_layers = 3

# training
num_epochs = 20
batch_size = 128
lr = 1e-4

# Define encode and decode as top-level functions
def encode(text, stoi):
    return [stoi[c] for c in text]

def decode(indices, itos):
    return ''.join([itos[i] for i in indices])

# Define get_batches function
def get_batches(X, y, batch_size, shuffle=True):
    """Generate batches of data."""
    if shuffle:
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        indices = mx.array(indices.tolist(), dtype=mx.int32)
        X = X[indices]
        y = y[indices]

    for i in range(0, X.shape[0], batch_size):
        X_batch = X[i:i + batch_size]
        y_batch = y[i:i + batch_size]
        yield X_batch, y_batch

### Data Loading and Tokenization
def load_or_process_data():
    processed_data_path = 'processed_data.pkl'

    if os.path.exists(processed_data_path):
        try:
            print("Loading preprocessed data...")
            with open(processed_data_path, 'rb') as f:
                return pickle.load(f)
        except (EOFError, pickle.UnpicklingError):
            print("Error: 'processed_data.pkl' is corrupted. Deleting the file and regenerating.")
            os.remove(processed_data_path)

    print("Starting data loading and tokenization...")
    def load_parquet_data(folder_path):
        all_texts = []
        for file_path in glob(os.path.join(folder_path, '*.parquet')):
            print(f"Loading file: {file_path}")
            table = pq.read_table(file_path)
            texts = table['text'].to_pylist()
            all_texts.extend(texts)
        return ' '.join(all_texts)

    text = load_parquet_data('./dataset')
    print(f"Loaded {len(text)} characters from Parquet files")

    vocab = sorted(list(set(text)))
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")

    print("Creating encoding and decoding functions...")
    itos = {i: c for i, c in enumerate(vocab)}
    stoi = {c: i for i, c in enumerate(vocab)}

    print("Encoding data...")
    data = encode(text, stoi)
    print("Splitting data into train and validation sets...")
    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    print("Preparing training and validation data...")
    X_train = mx.array([train_data[i:i + ctx_len] for i in range(0, len(train_data) - ctx_len, ctx_len)], dtype=mx.int32)
    y_train = mx.array([train_data[i + 1:i + ctx_len + 1] for i in range(0, len(train_data) - ctx_len, ctx_len)], dtype=mx.int32)
    X_val = mx.array([val_data[i:i + ctx_len] for i in range(0, len(val_data) - ctx_len, ctx_len)], dtype=mx.int32)
    y_val = mx.array([val_data[i + 1:i + ctx_len + 1] for i in range(0, len(val_data) - ctx_len, ctx_len)], dtype=mx.int32)

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

    processed_data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }

    print("Saving processed data...")
    with open(processed_data_path, 'wb') as f:
        pickle.dump(processed_data, f)
        f.flush()
        os.fsync(f.fileno())

    return processed_data

# Load or process data
data = load_or_process_data()
X_train, y_train = data['X_train'], data['y_train']
X_val, y_val = data['X_val'], data['y_val']
vocab_size = data['vocab_size']
itos, stoi = data['itos'], data['stoi']

### Block Definition
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.k_proj = nn.Linear(n_emb, head_size, bias=False)
        self.q_proj = nn.Linear(n_emb, head_size, bias=False)
        self.v_proj = nn.Linear(n_emb, head_size, bias=False)
        indices = mx.arange(ctx_len)
        mask = indices[:, None] < indices[None]
        self._causal_mask = mask * -1e9
        self.c_proj = nn.Linear(head_size, n_emb)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
    def __call__(self, x):
        B, T, C = x.shape
        K = self.k_proj(x)
        Q = self.q_proj(x)
        V = self.v_proj(x)
        mha_shape = (B, T, n_heads, head_size // n_heads)
        K = mx.as_strided(K, (mha_shape)).transpose([0, 2, 1, 3])
        Q = mx.as_strided(Q, (mha_shape)).transpose([0, 2, 1, 3])
        V = mx.as_strided(V, (mha_shape)).transpose([0, 2, 1, 3])
        attn_weights = (Q @ K.transpose([0, 1, 3, 2])) / math.sqrt(Q.shape[-1])
        attn_weights = attn_weights + self._causal_mask[:T, :T]
        attn_weights = mx.softmax(attn_weights, axis=-1)
        attn_weights = self.attn_dropout(attn_weights)
        o = (attn_weights @ V)
        o = o.transpose([0, 2, 1, 3]).reshape((B, T, head_size))
        o = self.c_proj(self.resid_dropout(o))
        return o

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(n_emb, 4 * n_emb)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_emb, n_emb)
        self.dropout = nn.Dropout(dropout)
    def __call__(self, x):
        x = self.gelu(self.c_fc(x))
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = MLP()
        self.mha = MultiHeadAttention()
        self.ln_1 = nn.LayerNorm(dims=n_emb)
        self.ln_2 = nn.LayerNorm(dims=n_emb)
    def __call__(self, x):
        x = x + self.mha(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

### Model Definition
class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_emb)
        self.wpe = nn.Embedding(ctx_len, n_emb)
        self.blocks = nn.Sequential(
            *[Block() for _ in range(n_layers)],
        )
        self.ln_f = nn.LayerNorm(dims=n_emb)
        self.lm_head = nn.Linear(n_emb, vocab_size)
        self._init_parameters()
    def __call__(self, x):
        B, T = x.shape
        tok_emb = self.wte(x)
        pos_emb = self.wpe(mx.arange(T))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
    def generate(self, max_new_tokens):
        ctx = mx.zeros((1, 1), dtype=mx.int32)
        for _ in range(max_new_tokens):
            logits = self(ctx[:, -ctx_len:])
            logits = logits[:, -1, :]
            next_tok = mx.random.categorical(logits, num_samples=1)
            ctx = mx.concatenate((ctx, next_tok), axis=1)
        return ctx
    def _init_parameters(self):
        normal_init = nn.init.normal(mean=0.0, std=0.02)
        residual_init = nn.init.normal(mean=0.0, std=(0.02 / math.sqrt(2 * n_layers)))
        new_params = []
        for name, module in self.named_modules():
            if isinstance(module, nn.layers.linear.Linear):
                if 'c_proj' in name:
                    new_params.append((name + '.weight', residual_init(module.weight)))
                else:
                    new_params.append((name + '.weight', normal_init(module.weight)))
                if 'bias' in module:
                    new_params.append((name + '.bias', mx.zeros(module.bias.shape)))
            elif isinstance(module, nn.layers.embedding.Embedding):
                new_params.append((name + '.weight', normal_init(module.weight)))
        self = self.update(utils.tree_unflatten(new_params))

### Training
print("Starting model training...")
# Ensure models directory exists
if not os.path.exists('models'):
    os.makedirs('models')
    print("Created 'models' directory")

def loss_fn(model, x, y):
    logits = model(x)
    B, T, C = logits.shape
    logits = logits.reshape(B * T, C)
    y = y.reshape(B * T)
    loss = nn.losses.cross_entropy(logits, y, reduction='mean')
    return loss

print("Initializing model...")
model = GPT()

# Load the latest checkpoint if it exists
checkpoints = sorted(glob('models/checkpoint_epoch_*.pkl'))
start_epoch = 0
start_batch = 0  # Initialize starting batch to 0
if checkpoints:
    latest_checkpoint = checkpoints[-1]
    print(f"Loading checkpoint: {latest_checkpoint}")
    with open(latest_checkpoint, 'rb') as f:
        checkpoint = pickle.load(f)
    model.update(checkpoint['model_state'])
    start_epoch = checkpoint['epoch']
    start_batch = checkpoint.get('batch', 0)  # Use .get() to handle missing 'batch' field
    print(f"Resuming from epoch {start_epoch} batch {start_batch}")
else:
    print("No checkpoint found, starting from scratch")

mx.eval(model.parameters())
loss_and_grad = nn.value_and_grad(model, loss_fn)
optimizer = optim.AdamW(learning_rate=lr)

# Initialize avg_train_loss and avg_val_loss for exception handling
avg_train_loss = None
avg_val_loss = None

print(f"Starting training for {num_epochs} epochs...")

try:
    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train(True)
        running_loss = 0
        batch_cnt = 0

        # Skip the batches that were already processed in the current epoch
        for input, label in get_batches(X_train, y_train, batch_size):
            batch_cnt += 1
            if batch_cnt <= start_batch:
                continue  # Skip processed batches

            # Start timing the batch
            start_time = time.time()

            loss, grads = loss_and_grad(model, input, label)
            optimizer.update(model, grads)
            running_loss += loss.item()

            # End timing the batch
            end_time = time.time()
            time_per_batch = end_time - start_time

            # Print the result in the format "Batch = X | Time = Y seconds | loss = Z"
            if batch_cnt % 100 == 0:
                print(f"Batch = {batch_cnt} | loss = {loss.item():.4f}")

            mx.eval(model.parameters(), optimizer.state)
        avg_train_loss = running_loss / batch_cnt

        # Validation step
        print("Evaluating on validation set...")
        model.train(False)
        running_loss = 0
        batch_cnt = 0
        for input, label in get_batches(X_val, y_val, batch_size, shuffle=False):
            batch_cnt += 1
            loss = loss_fn(model, input, label)
            running_loss += loss.item()
        avg_val_loss = running_loss / batch_cnt
        print(f"Epoch {epoch+1:2} | train = {avg_train_loss:.4f} | val = {avg_val_loss:.4f}")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'batch': 0,  # Reset batch to 0 for the next epoch
            'model_state': model.parameters(),
            'optimizer_state': optimizer.state,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        }
        checkpoint_path = os.path.join('models', f'checkpoint_epoch_{epoch+1}.pkl')
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        print(f"Saved checkpoint to {checkpoint_path}")
        start_batch = 0  # Reset start_batch for the next epoch

except KeyboardInterrupt:
    print("Training interrupted. Saving checkpoint...")
    checkpoint = {
        'epoch': epoch,
        'batch': batch_cnt,  # Save the current batch count
        'model_state': model.parameters(),
        'optimizer_state': optimizer.state,
        'train_loss': avg_train_loss if avg_train_loss is not None else 0.0,
        'val_loss': avg_val_loss if avg_val_loss is not None else 0.0
    }
    checkpoint_path = os.path.join('models', f'checkpoint_epoch_{epoch+1}_interrupt.pkl')
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"Saved checkpoint to {checkpoint_path} on interrupt.")

print("Training completed. Starting inference...")

# Inference
print("Generating text based on initial content...")

# Init info
initial_text = "Where is Zuckerberg? "
initial_indices = encode(initial_text, stoi) 

ctx = mx.array([initial_indices], dtype=mx.int32)


def generate_with_initial_context(model, max_new_tokens, ctx):
    for _ in range(max_new_tokens):
        logits = model(ctx[:, -ctx_len:])
        logits = logits[:, -1, :]
        next_token = mx.random.categorical(logits, num_samples=1)
        

        ctx = mx.concatenate((ctx, next_token), axis=1)
    
    return ctx


generated_indices = generate_with_initial_context(model, 1000, ctx)[0].tolist()


completion = decode(generated_indices, itos)


print("Generated text based on the initial content:")
print(completion)


with open('completions.txt', 'w') as f:
    f.write(completion)

print("Generated text saved to completions.txt")