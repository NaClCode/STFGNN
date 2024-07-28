import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from utils_4n0_3layer_12T_res import generate_data, masked_mae_np, masked_mape_np, masked_mse_np, construct_model
from model.stsgcn_4n_res import huber_loss

def train_model(model, config, train_loader, val_loader, test_loader, val_y, test_y, device):
    
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    lr_scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: (1 - epoch / config['epochs']) ** 2
    )

    global_epoch = 1
    epochs = config['epochs']
    best_val_loss = float('inf')
    all_info = []

    for epoch in range(epochs):
        model.train()
        epoch_start_time = time.time()
        train_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = huber_loss(prediction, y).sum()
            loss.backward(retain_graph=True)
            optimizer.step()
            train_loss += loss.item()
            

        train_loss /= len(train_loader)

        model.eval()
        with torch.no_grad():
            val_loss = 0
            predictions = []

            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                prediction = model(x)
                val_loss += model.loss(prediction, y).item()
                predictions.append(prediction.cpu().numpy())

            val_loss /= len(val_loader)
            predictions = np.concatenate(predictions, axis=0)

        val_mae = masked_mae_np(val_y, predictions, 0)

        print(f'Epoch {global_epoch}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_mae={val_mae:.4f}, time={time.time() - epoch_start_time:.2f}s')

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            test_predictions = []
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                prediction = model(x)
                test_predictions.append(prediction.cpu().numpy())

            test_predictions = np.concatenate(test_predictions, axis=0)
            tmp_info = []

            for i in range(config['num_for_predict']):
                y, x = test_y[:, : i + 1, :], test_predictions[:, : i + 1, :]
                tmp_info.append((
                    masked_mae_np(y, x, 0),
                    masked_mape_np(y, x, 0),
                    masked_mse_np(y, x, 0) ** 0.5
                ))

            mae, mape, rmse = tmp_info[-1]
            print(f'Test results: MAE={mae:.2f}, MAPE={mape:.2f}, RMSE={rmse:.2f}')

            all_info.append([global_epoch, train_loss, val_mae, mae, mape, rmse, tmp_info])

        lr_scheduler.step()
        global_epoch += 1

    best_epoch_info = min(all_info, key=lambda x: x[2])
    print(f'Best Epoch Info: {best_epoch_info}')

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help='Configuration file')
    parser.add_argument("--test", action="store_true", help="Test program")
    parser.add_argument("--save", action="store_true", help="Save model")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = construct_model(config).to(device)

    batch_size = config['batch_size']
    num_of_vertices = config['num_of_vertices']
    graph_signal_matrix_filename = config['graph_signal_matrix_filename']

    loaders = []
    true_values = []

    for idx, (x, y) in enumerate(generate_data(graph_signal_matrix_filename)):
        if args.test:
            x, y = x[:100], y[:100]

        y = y.squeeze(axis=-1)
        dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
        shuffle = (idx == 0)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        loaders.append(loader)

        if idx == 0:
            training_samples = x.shape[0]
        else:
            true_values.append(y)

    train_loader, val_loader, test_loader = loaders
    val_y, test_y = true_values
    
    model = train_model(model, config, train_loader, val_loader, test_loader, val_y, test_y, device)

    # if args.save:
    #     torch.save(model.state_dict(), 'STSGCN_model.pth')


if __name__ == "__main__":
    main()
