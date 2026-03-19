import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMForecaster, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out

def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data)-seq_length):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length][0] # predecir precio
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def main():
    print("--- Modelo LSTM (Deep Learning SOTA) ---")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "..", "data", "raw", "dataset_multivariante_completo.csv")
    
    df = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
    df = df.dropna()
    
    # Feature scaling (Vital para Redes Neuronales) 
    # Usamos Precio (target) y Demanda (exog)
    features = ['precio_mwh', 'demanda_real']
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features])
    
    seq_length = 96 # Miramos las ultimas 24h para predecir el proximo paso
    X, y = create_sequences(scaled_data, seq_length)
    
    # Train-test split
    n_test = 96 * 2 # ultimos 2 dias
    X_train, y_train = X[:-n_test], y[:-n_test]
    X_test, y_test = X[-n_test:], y[-n_test:]
    
    # Arrays a tensores
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float().unsqueeze(1)
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float().unsqueeze(1)
    
    # DataLoaders
    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    
    # Model
    model = LSTMForecaster(input_dim=2, hidden_dim=64, num_layers=2, output_dim=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("Entrenando LSTM (epocas=10)...")
    model.train()
    for epoch in range(10):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
    print("Prediciendo...")
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test).numpy()
        
    # Invertir el scaling solo para el target (columna 0)
    def invert_scale(data_1d):
        dummy = np.zeros((len(data_1d), 2))
        dummy[:, 0] = data_1d.flatten()
        return scaler.inverse_transform(dummy)[:, 0]
        
    preds_real = invert_scale(test_preds)
    y_test_real = invert_scale(y_test.numpy())
    
    mae = mean_absolute_error(y_test_real, preds_real)
    rmse = np.sqrt(mean_squared_error(y_test_real, preds_real))
    print(f"LSTM - MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    # Plot
    test_idx = df.index[seq_length + len(X_train):]
    plt.figure(figsize=(10,5))
    plt.plot(test_idx, y_test_real, label='Real')
    plt.plot(test_idx, preds_real, label='LSTM Predict', linestyle='dashed')
    plt.title("Predicción LSTM (Deep Learning SOTA)")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(script_dir, "lstm_plot.png"))
    print("Guardado en lstm_plot.png")

if __name__ == "__main__":
    main()
