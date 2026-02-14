# ============================================================
# ADVANCED TIME SERIES FORECASTING WITH CUSTOM TRANSFORMER
# ============================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import math
import random
import os

# ============================================================
# 0. REPRODUCIBILITY
# ============================================================

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 1. DATA GENERATION (MULTIVARIATE, MULTI-SEASONAL)
# ============================================================

def generate_multivariate_time_series(n_steps=3000, n_features=3):

    t = np.arange(n_steps)

    trend = 0.0008 * t

    daily = np.sin(2 * np.pi * t / 24)
    weekly = np.sin(2 * np.pi * t / (24 * 7))

    data = []

    for i in range(n_features):
        phase_shift = i * 3
        feature = (
            trend
            + (1.0 + 0.3 * i) * np.roll(daily, phase_shift)
            + (0.7 + 0.2 * i) * np.roll(weekly, phase_shift)
            + np.random.normal(0, 0.3, size=n_steps)
        )
        data.append(feature)

    data = np.stack(data, axis=1)
    return data


data = generate_multivariate_time_series()
pd.DataFrame(data).to_csv("generated_timeseries.csv", index=False)

# ============================================================
# 2. DATA PREPARATION
# ============================================================

INPUT_LEN = 96
OUTPUT_LEN = 24
BATCH_SIZE = 32


class TimeSeriesDataset(Dataset):
    def __init__(self, data, input_len, output_len):
        self.X = []
        self.Y = []

        for i in range(len(data) - input_len - output_len):
            self.X.append(data[i:i+input_len])
            self.Y.append(data[i+input_len:i+input_len+output_len])

        self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.Y = torch.tensor(np.array(self.Y), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


# Train / Val / Test Split
train_size = int(len(data) * 0.7)
val_size = int(len(data) * 0.15)

train_data = data[:train_size]
val_data = data[train_size:train_size+val_size]
test_data = data[train_size+val_size:]

scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
val_data = scaler.transform(val_data)
test_data = scaler.transform(test_data)

train_ds = TimeSeriesDataset(train_data, INPUT_LEN, OUTPUT_LEN)
val_ds = TimeSeriesDataset(val_data, INPUT_LEN, OUTPUT_LEN)
test_ds = TimeSeriesDataset(test_data, INPUT_LEN, OUTPUT_LEN)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

# ============================================================
# 3. CUSTOM TRANSFORMER IMPLEMENTATION
# ============================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

        self.attention_weights = None

    def forward(self, query, key, value):

        N = query.shape[0]

        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        Q = Q.view(N, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(N, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(N, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attention = torch.softmax(scores, dim=-1)

        self.attention_weights = attention.detach().cpu()

        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(N, -1, self.d_model)

        return self.fc_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, heads, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.ReLU(),
            nn.Linear(d_model*4, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)


class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, heads, dropout)
            for _ in range(layers)
        ])
        self.output_proj = nn.Linear(d_model, input_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x)
        return self.output_proj(x[:, -OUTPUT_LEN:])


# ============================================================
# 4. HYPERPARAMETER TUNING
# ============================================================

configs = [
    {"d_model": 64, "heads": 4, "lr": 1e-3},
    {"d_model": 128, "heads": 4, "lr": 5e-4},
    {"d_model": 64, "heads": 8, "lr": 1e-3},
]

best_rmse = float("inf")
best_model = None

for config in configs:
    print(f"\nTraining config: {config}")

    model = TransformerModel(
        input_dim=3,
        d_model=config["d_model"],
        heads=config["heads"]
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.MSELoss()

    for epoch in range(10):
        model.train()
        for X, Y in train_loader:
            X, Y = X.to(DEVICE), Y.to(DEVICE)
            output = model(X)
            loss = criterion(output, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Validation
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for X, Y in val_loader:
            X, Y = X.to(DEVICE), Y.to(DEVICE)
            output = model(X)
            preds.append(output.cpu().numpy())
            actuals.append(Y.cpu().numpy())

    preds = np.concatenate(preds)
    actuals = np.concatenate(actuals)
    rmse = np.sqrt(mean_squared_error(actuals.flatten(), preds.flatten()))
    print("Validation RMSE:", rmse)

    if rmse < best_rmse:
        best_rmse = rmse
        best_model = model

model = best_model

# ============================================================
# 5. TEST EVALUATION
# ============================================================

model.eval()
preds, actuals = [], []

with torch.no_grad():
    for X, Y in test_loader:
        X, Y = X.to(DEVICE), Y.to(DEVICE)
        output = model(X)
        preds.append(output.cpu().numpy())
        actuals.append(Y.cpu().numpy())

preds = np.concatenate(preds)
actuals = np.concatenate(actuals)

transformer_rmse = np.sqrt(mean_squared_error(actuals.flatten(), preds.flatten()))
transformer_mae = mean_absolute_error(actuals.flatten(), preds.flatten())

# ============================================================
# 6. XGBOOST BASELINE
# ============================================================

def create_lag_features(data, lags=INPUT_LEN):
    X, Y = [], []
    for i in range(len(data) - lags - OUTPUT_LEN):
        X.append(data[i:i+lags].flatten())
        Y.append(data[i+lags:i+lags+OUTPUT_LEN].flatten())
    return np.array(X), np.array(Y)

X_train, Y_train = create_lag_features(train_data)
X_test, Y_test = create_lag_features(test_data)

xgb_model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    objective="reg:squarederror"
)

xgb_model.fit(X_train, Y_train)
xgb_preds = xgb_model.predict(X_test)

xgb_rmse = np.sqrt(mean_squared_error(Y_test.flatten(), xgb_preds.flatten()))
xgb_mae = mean_absolute_error(Y_test.flatten(), xgb_preds.flatten())

print("\n================ RESULTS ================")
print("Transformer RMSE:", transformer_rmse)
print("Transformer MAE :", transformer_mae)
print("XGBoost RMSE    :", xgb_rmse)
print("XGBoost MAE     :", xgb_mae)

# ============================================================
# 7. EXTRACT TRUE ATTENTION WEIGHTS
# ============================================================

sample_X, _ = test_ds[0]
sample_X = sample_X.unsqueeze(0).to(DEVICE)

with torch.no_grad():
    _ = model(sample_X)

attention_matrix = model.layers[0].attn.attention_weights[0][0].numpy()
np.savetxt("attention_weights.csv", attention_matrix, delimiter=",")

print("Attention weights saved to attention_weights.csv")
