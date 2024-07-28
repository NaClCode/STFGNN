import numpy as np
import pandas as pd
import os
import torch
# 构建模型
def construct_model(config):
    from model.stsgcn_4n_res import STSGCN

    # 从配置字典中获取各个参数
    module_type = config['module_type']
    act_type = config['act_type']
    temporal_emb = config['temporal_emb']
    spatial_emb = config['spatial_emb']
    use_mask = config['use_mask']
    batch_size = config['batch_size']

    num_of_vertices = config['num_of_vertices']
    num_of_features = config['num_of_features']
    points_per_hour = config['points_per_hour']
    num_for_predict = config['num_for_predict']
    adj_filename = config['adj_filename']
    id_filename = config['id_filename']
    if id_filename is not None:
        if not os.path.exists(id_filename):
            id_filename = None

    # 获取邻接矩阵
    adj = get_adjacency_matrix(adj_filename, num_of_vertices, id_filename=id_filename)
    # 读取DTW邻接矩阵
    adj_dtw = np.array(pd.read_csv(config['adj_dtw_filename'], header=None))
    # 构建融合的邻接矩阵
    adj_mx = construct_adj_fusion(adj, adj_dtw, 4)
    print("The shape of localized adjacency matrix: {}".format(adj_mx.shape), flush=True)

    filters = config['filters']
    first_layer_embedding_size = config['first_layer_embedding_size']

    adj_mx = torch.tensor(adj_mx, dtype=torch.float32)

    mask_init_value = (adj_mx != 0).float()
  
    # 构建模型
    net = STSGCN(
        adj_mx,
        points_per_hour, num_of_vertices, num_of_features,first_layer_embedding_size, 
        filters, module_type, act_type,
        use_mask, mask_init_value, temporal_emb, spatial_emb,
        prefix="", predict_length=num_for_predict
    )
    
    return net

def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
    import csv
    A = np.zeros((num_of_vertices, num_of_vertices), dtype=np.float32)
    if id_filename:
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}
        with open(distance_df_filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                A[id_dict[i], id_dict[j]] = 1
                A[id_dict[j], id_dict[i]] = 1
        return A

    with open(distance_df_filename, 'r') as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            i, j, distance = int(row[0]), int(row[1]), float(row[2])
            A[i, j] = 1
            A[j, i] = 1
    return A

def construct_adj_fusion(A, A_dtw, steps):
    N = len(A)
    adj = np.zeros([N * steps] * 2)
    for i in range(steps):
        if i in [1, 2]:
            adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A
        else:
            adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A_dtw
    for i in range(N):
        for k in range(steps - 1):
            adj[k * N + i, (k + 1) * N + i] = 1
            adj[(k + 1) * N + i, k * N + i] = 1
    adj[3 * N: 4 * N, 0: N] = A_dtw
    adj[0: N, 3 * N: 4 * N] = A_dtw
    adj[2 * N: 3 * N, 0: N] = adj[0 * N: 1 * N, 1 * N: 2 * N]
    adj[0: N, 2 * N: 3 * N] = adj[0 * N: 1 * N, 1 * N: 2 * N]
    adj[1 * N: 2 * N, 3 * N: 4 * N] = adj[0 * N: 1 * N, 1 * N: 2 * N]
    adj[3 * N: 4 * N, 1 * N: 2 * N] = adj[0 * N: 1 * N, 1 * N: 2 * N]
    for i in range(len(adj)):
        adj[i, i] = 1
    return adj

def generate_data(graph_signal_matrix_filename):
    data = np.load(graph_signal_matrix_filename, allow_pickle=True)
    keys = data.files
    if 'train' in keys and 'val' in keys and 'test' in keys:
        yield from generate_from_train_val_test(data)
    elif 'data' in keys:
        length = data['data'].shape[0]
        yield from generate_from_data(data, length)
    else:
        raise KeyError("Data not found in the file")

def generate_from_train_val_test(data):
    mean = None
    std = None
    for key in ('train', 'val', 'test'):
        x, y = generate_seq(data[key], 12, 12)
        if mean is None:
            mean = x.mean()
        if std is None:
            std = x.std()
        yield (x - mean) / std, y

def generate_from_data(data, length):
    mean = None
    std = None
    train_line, val_line = int(length * 0.6), int(length * 0.8)
    for line1, line2 in ((0, train_line), (train_line, val_line), (val_line, length)):
        x, y = generate_seq(data['data'][line1: line2], 12, 12)
        if mean is None:
            mean = x.mean()
        if std is None:
            std = x.std()
        yield (x - mean) / std, y

def generate_seq(data, train_length, pred_length):
    seq = np.concatenate([np.expand_dims(data[i: i + train_length + pred_length], 0)
                          for i in range(data.shape[0] - train_length - pred_length + 1)],
                         axis=0)[:, :, :, 0: 1]
    return np.split(seq, 2, axis=1)

def mask_np(array, null_val):
    if np.isnan(null_val):
        return (~np.isnan(null_val)).astype('float32')
    else:
        return np.not_equal(array, null_val).astype('float32')

def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = mask_np(y_true, null_val)
        mask /= mask.mean()
        mape = np.abs((y_pred - y_true) / y_true)
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100

def masked_mse_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mse = (y_true - y_pred) ** 2
    return np.mean(np.nan_to_num(mask * mse))

def masked_mae_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mae = np.abs(y_true - y_pred)
    return np.mean(np.nan_to_num(mask * mae))
