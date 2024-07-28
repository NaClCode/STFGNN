import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionEmbedding(nn.Module):
    def __init__(self, input_length, num_of_vertices, embedding_size, temporal=True, spatial=True, prefix=""):
        super(PositionEmbedding, self).__init__()
        self.temporal = temporal
        self.spatial = spatial
        self.temporal_emb = None
        self.spatial_emb = None
        
        if temporal:
            self.temporal_emb = nn.Parameter(torch.randn(1, input_length, 1, embedding_size))
        if spatial:
            self.spatial_emb = nn.Parameter(torch.randn(1, 1, num_of_vertices, embedding_size))

    def forward(self, data):
        if self.temporal_emb is not None:
            data = data + self.temporal_emb
        if self.spatial_emb is not None:
            data = data + self.spatial_emb
        return data


class GCNOperation(nn.Module):
    def __init__(self, adj, num_of_filter, num_of_features, activation):
        super(GCNOperation, self).__init__()
        self.adj = adj
        self.num_of_filter = num_of_filter
        self.num_of_features = num_of_features
        assert activation in {'GLU', 'relu'}
        self.activation = activation

        if activation == 'GLU':
            self.fc = nn.Linear(num_of_features, 2 * num_of_filter)
        else:
            self.fc = nn.Linear(num_of_features, num_of_filter)

    def forward(self, data):
        # (4N, B, C)
        data = torch.permute(data, (1, 0, 2))
        # (B, 4N, C)
        data = self.adj @ data
        # (B, 4N, C)
        data = torch.permute(data, (1, 0, 2))

        if self.activation == 'GLU':
            data = self.fc(data)
            lhs, rhs = torch.split(data, self.num_of_filter, dim=-1)
            return lhs * torch.sigmoid(rhs)
        elif self.activation == 'relu':
            return F.relu(self.fc(data))


class STSGCM(nn.Module):
    def __init__(self, adj, filters, num_of_features, num_of_vertices, activation):
        super(STSGCM, self).__init__()
        self.adj = adj
        self.num_of_features = num_of_features
        self.num_of_vertices = num_of_vertices
        self.activation = activation

        self.gcns = nn.ModuleList([GCNOperation(adj, f, num_of_features if i == 0 else filters[i - 1], activation) for i, f in enumerate(filters)])

    def forward(self, data):
        need_concat = []
        for gcn in self.gcns:
            data = gcn(data)
            need_concat.append(data)
        
        need_concat = [t[self.num_of_vertices:2*self.num_of_vertices, :, :].unsqueeze(0) for t in need_concat]
        return torch.max(torch.cat(need_concat, dim=0), dim=0)[0]


class STHGNNLayerIndividual(nn.Module):
    def __init__(self, adj, T, num_of_vertices, num_of_features, filters, activation, temporal_emb=True, spatial_emb=True, prefix=""):
        super(STHGNNLayerIndividual, self).__init__()
        self.position_embedding = PositionEmbedding(T, num_of_vertices, num_of_features, temporal_emb, spatial_emb, prefix)
        self.adj = adj
        self.T = T
        self.num_of_vertices = num_of_vertices
        self.num_of_features = num_of_features
        self.filters = filters
        self.activation = activation

        self.temporal_conv_left = nn.Conv2d(in_channels=num_of_features, out_channels=num_of_features, kernel_size=(1, 2), stride=(1, 1), dilation=(1, 3))
        self.temporal_conv_right = nn.Conv2d(in_channels=num_of_features, out_channels=num_of_features, kernel_size=(1, 2), stride=(1, 1), dilation=(1, 3))

        self.stsgcms = nn.ModuleList([STSGCM(adj, filters, num_of_features, num_of_vertices, activation) for _ in range(T-3)])

    def forward(self, data):
        data = self.position_embedding(data)
        data_temp = data.permute(0, 3, 2, 1)  # (B, C, N, T)
        data_left = torch.sigmoid(self.temporal_conv_left(data_temp))
        data_right = torch.tanh(self.temporal_conv_right(data_temp))
        data_time_axis = data_left * data_right
        data_res = data_time_axis.permute(0, 3, 2, 1)  # (B, T-3, N, C)
       
        need_concat = []
        for i, stsgcm in enumerate(self.stsgcms):
            t = data[:, i:i+4, :, :].reshape(-1, 4 * self.num_of_vertices, self.num_of_features).permute(1, 0, 2)  # (4N, B, C)
            t = stsgcm(t)
            t = t.permute(1, 0, 2)  # (B, N, C')
            need_concat.append(t.unsqueeze(1))  # (B, 1, N, C')

        need_concat = torch.cat(need_concat, dim=1)  # (B, T-3, N, C')
        return need_concat + data_res


class OutputLayer(nn.Module):
    def __init__(self, num_of_vertices, input_length, num_of_features, num_of_filters=128, predict_length=12):
        super(OutputLayer, self).__init__()
        self.num_of_vertices = num_of_vertices
        self.input_length = input_length
        self.num_of_features = num_of_features
        self.num_of_filters = num_of_filters
        self.predict_length = predict_length

        self.fc1 = nn.Linear(input_length * num_of_features, num_of_filters)
        self.fc2 = nn.Linear(num_of_filters, predict_length)

    def forward(self, data):
        data = data.permute(0, 2, 1, 3).reshape(-1, self.num_of_vertices, self.input_length * self.num_of_features)
        data = F.relu(self.fc1(data))
        data = self.fc2(data)
        return data.permute(0, 2, 1)  # (B, T', N)


def huber_loss(data, label, rho=1):
    loss = torch.abs(data - label)
    loss = torch.where(loss > rho, loss - 0.5 * rho, (0.5 / rho) * torch.square(loss))
    return loss


def weighted_loss(data, label, input_length, rho=1):
    weight = torch.flip(torch.arange(1, input_length + 1, dtype=torch.float32), dims=[0]).unsqueeze(0).unsqueeze(-1)
    return huber_loss(data, label, rho) * weight


class STSGCN(nn.Module):
    def __init__(self, adj, input_length, num_of_vertices, num_of_features_first, num_of_features, filter_list, module_type, activation, use_mask=True, mask_init_value=None, temporal_emb=True, spatial_emb=True, prefix="", predict_length=12):
        super(STSGCN, self).__init__()
        self.use_mask = bool(use_mask)
        if use_mask:
            self.mask = nn.Parameter(mask_init_value)
            self.adj = self.mask * adj
        else:
            self.adj = adj
        
        self.stsgcls = nn.ModuleList()
        for idx, filters in enumerate(filter_list):
            self.stsgcls.append(STHGNNLayerIndividual(self.adj, input_length, num_of_vertices, num_of_features, filters, activation, temporal_emb, spatial_emb, prefix=f"{prefix}_stsgcl_{idx}"))
            input_length -= 3
            num_of_features = filters[-1]
   
        self.first = nn.Linear(in_features=num_of_features_first, out_features=num_of_features)
        self.output_layers = nn.ModuleList([OutputLayer(num_of_vertices, input_length, num_of_features, num_of_filters=128, predict_length=1) for _ in range(predict_length)])

    def forward(self, data):
        B, T, N, C = data.shape

        data = data.view(B * T * N, C)
        data = self.first(data)
        data = data.view(B, T, N, -1)

        for stsgcl in self.stsgcls:
            data = stsgcl(data)

        outputs = []
        for output_layer in self.output_layers:
            outputs.append(output_layer(data))

        data = torch.cat(outputs, dim=1)
        
        return data

# Example usage
# Define your data and adjacency matrix
# data = torch.randn(batch_size, T, N, C)
# adj = torch.randn(4 * N, 4 * N)
# label = torch.randn(batch_size, T, N)
# model = STSGCN(adj, input_length=T, num_of_vertices=N, num_of_features=C, filter_list=[[64, 32], [32, 16]], module_type='individual', activation='GLU')
# loss, predictions = model(data, label)
