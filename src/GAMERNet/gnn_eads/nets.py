"""Module containing the Graph Neural Network classes."""
from typing import Optional
import math

import torch
from torch.nn import Linear, Softplus
from torch import Tensor
from torch.distributions import Normal
from torch_geometric.nn.conv import SAGEConv, GATConv, GATv2Conv
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn.aggr import GraphMultisetTransformer

class MAB(torch.nn.Module):
    def __init__(self, dim_Q: int, dim_K: int, dim_V: int, num_heads: int, bias: bool):
        """
        Multihead Attention Block (MAB) module.
        Readapted from PyG 2.0.3 without normalization and dropout.
        """
        super().__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads

        self.fc_q = Linear(dim_Q, dim_V, bias)
        self.layer_k = Linear(dim_K, dim_V, bias)
        self.layer_v = Linear(dim_K, dim_V, bias)
        self.fc_o = Linear(dim_V, dim_V, bias)

    def reset_parameters(self):
        self.fc_q.reset_parameters()
        self.layer_k.reset_parameters()
        self.layer_v.reset_parameters()
        self.fc_o.reset_parameters()
        pass

    def forward(
        self,
        Q: Tensor,
        K: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:

        Q = self.fc_q(Q)

        K, V = self.layer_k(K), self.layer_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), dim=0)
        K_ = torch.cat(K.split(dim_split, 2), dim=0)
        V_ = torch.cat(V.split(dim_split, 2), dim=0)

        if mask is not None:
            mask = torch.cat([mask for _ in range(self.num_heads)], 0)
            attention_score = Q_.bmm(K_.transpose(1, 2))
            attention_score = attention_score / math.sqrt(self.dim_V)
            A = torch.softmax(mask + attention_score, 1)
        else:
            A = torch.softmax(
                Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 1)

        out = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)

        out = out + self.fc_o(out).relu()        

        return out

class PMA(torch.nn.Module):
    def __init__(self, channels: int, num_heads: int, num_seeds: int, bias: bool):
        super().__init__()
        self.S = torch.nn.Parameter(torch.Tensor(1, num_seeds, channels))
        self.mab = MAB(channels, channels, channels, num_heads, bias)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.S)
        self.mab.reset_parameters()

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        return self.mab(self.S.repeat(x.size(0), 1, 1), x, mask)
    
    
class FlexibleNet(torch.nn.Module):
    def __init__(self, 
                 in_features: int,                 
                 dim: int=128,                  
                 N_linear: int=0,
                 N_conv: int=3,
                 adj_conv: bool=False,
                 sigma=torch.nn.ReLU(),
                 bias: bool=True,
                 conv=SAGEConv,
                 pool=GraphMultisetTransformer, 
                 pool_ratio: float=0.25, 
                 pool_heads: int=4, 
                 pool_seq: list[str]=["GMPool_G", "SelfAtt", "GMPool_I"], 
                 pool_layer_norm: bool=False):
        """
        Flexible Net for defining multiple GNN model architectures.

        Args:
            dim (int, optional): Layer width. Defaults to 128.
            N_linear (int, optional): Number of dense. Default to 3.
            N_conv (int, optional): Number of convolutional layers. Default to 3.
            adj_conv (bool, optional): Whether include linear layer between each convolution. Default to True.
            in_features (int, optional): Input graph node dimensionality. Default to NODE_FEATURES.
            sigma (_type_, optional): Activation function. Default to torch.nn.ReLU().
            bias (bool, optional): Bias inclusion. Default to True.
            conv (_type_, optional): Convolutional Layer. Default to SAGEConv.
            pool (_type_, optional): Pooling Layer. Default to GraphMultisetTransformer.
        """
        super(FlexibleNet, self).__init__()
        self.dim = dim
        self.in_features = in_features
        self.sigma = sigma
        self.conv = conv
        self.num_conv_layers = N_conv
        self.num_linear_layers = N_linear
        self.adj_conv = adj_conv        
        # Building blocks of the GNN
        self.input_layer = Linear(self.in_features, self.dim, bias=bias)
        self.linear_block = torch.nn.ModuleList([Linear(self.dim, self.dim, bias=bias) for _ in range(self.num_linear_layers)])
        self.conv_block = torch.nn.ModuleList([conv(self.dim, self.dim, bias=bias) for _ in range(self.num_conv_layers)])
        if self.adj_conv:
            self.adj_block = torch.nn.ModuleList([Linear(self.dim, self.dim, bias=bias) for _ in range(self.num_conv_layers)])
        self.pool = pool(self.dim, self.dim, 1, num_nodes=300,         
                         pooling_ratio=pool_ratio, pool_sequences=pool_seq,
                         num_heads=pool_heads, layer_norm=pool_layer_norm)                                                             
        
    def forward(self, data):
        #------------#
        # NODE LEVEL #
        #------------#        
        out = self.sigma(self.input_layer(data.x))  # Input layer
        # Dense layers 
        for layer in range(self.num_linear_layers):
            out = self.sigma(self.linear_block[layer](out))
        # Convolutional layers
        for layer in range(self.num_conv_layers):
            if self.adj_conv:
                out = self.sigma(self.adj_block[layer](out))
            out = self.sigma(self.conv_block[layer](out, data.edge_index))
        #-----------------------#
        # GRAPH LEVEL (POOLING) #
        #-----------------------#
        out = self.pool(out, data.batch, data.edge_index)
        return out.view(-1)
    

class GameNet(torch.nn.Module):
    def __init__(self, 
                 features_list: list[str],
                 scaling_params: dict,                
                 dim: int=128,                  
                 N_linear: int=0,
                 N_conv: int=3,
                 adj_conv: bool=False,
                 sigma=torch.nn.ReLU(),
                 bias: bool=True,
                 conv=SAGEConv,
                 pool=GraphMultisetTransformer, 
                 pool_ratio: float=0.25, 
                 pool_heads: int=4, 
                 pool_seq: list[str]=["GMPool_G", "SelfAtt", "GMPool_I"], 
                 pool_layer_norm: bool=False):
        """
        Flexible Net for defining multiple GNN model architectures.

        Args:
            dim (int, optional): Layer width. Defaults to 128.
            N_linear (int, optional): Number of dense. Default to 3.
            N_conv (int, optional): Number of convolutional layers. Default to 3.
            adj_conv (bool, optional): Whether include linear layer between each convolution. Default to True.
            in_features (int, optional): Input graph node dimensionality. Default to NODE_FEATURES.
            sigma (_type_, optional): Activation function. Default to torch.nn.ReLU().
            bias (bool, optional): Bias inclusion. Default to True.
            conv (_type_, optional): Convolutional Layer. Default to SAGEConv.
            pool (_type_, optional): Pooling Layer. Default to GraphMultisetTransformer.
        """
        super().__init__()
        self.dim = dim
        self.node_features_list = features_list
        self.in_features = len(features_list)
        self.scaling_params = scaling_params
        self.sigma = sigma
        self.conv = conv
        self.num_conv_layers = N_conv
        self.num_linear_layers = N_linear
        self.adj_conv = adj_conv        
        # Building blocks of the GNN
        self.input_layer = Linear(self.in_features, self.dim, bias=bias)
        self.linear_block = torch.nn.ModuleList([Linear(self.dim, self.dim, bias=bias) for _ in range(self.num_linear_layers)])
        self.conv_block = torch.nn.ModuleList([conv(self.dim, self.dim, bias=bias) for _ in range(self.num_conv_layers)])
        if self.adj_conv:
            self.adj_block = torch.nn.ModuleList([Linear(self.dim, self.dim, bias=bias) for _ in range(self.num_conv_layers)])
        self.pool = pool(self.dim, self.dim, 1, num_nodes=300,         
                         pooling_ratio=pool_ratio, pool_sequences=pool_seq,
                         num_heads=pool_heads, layer_norm=pool_layer_norm)                                                             
        
    def forward(self, data):
        #------------#
        # NODE LEVEL #
        #------------#        
        out = self.sigma(self.input_layer(data.x))  # Input layer
        # Dense layers 
        for layer in range(self.num_linear_layers):
            out = self.sigma(self.linear_block[layer](out))
        # Convolutional layers
        for layer in range(self.num_conv_layers):
            if self.adj_conv:
                out = self.sigma(self.adj_block[layer](out))
            out = self.sigma(self.conv_block[layer](out, data.edge_index))
        #-----------------------#
        # GRAPH LEVEL (POOLING) #
        #-----------------------#
        out = self.pool(out, data.batch)
        return out.view(-1)
    

class TestNet(torch.nn.Module):
    def __init__(self, 
                 features_list: list[str],
                 scaling_params: dict,                
                 dim: int,                  
                 N_linear: int=0,
                 N_conv: int=3,
                 adj_conv: bool=False,
                 sigma=torch.nn.ReLU(),
                 bias: bool=True,
                 conv=SAGEConv, 
                 pool_heads: int=4):
        """
        Flexible Net for defining multiple GNN model architectures.

        Args:
            dim (int, optional): Layer width.
            N_linear (int, optional): Number of dense. Default to 3.
            N_conv (int, optional): Number of convolutional layers. Default to 3.
            adj_conv (bool, optional): Whether include linear layer between each convolution. Default to True.
            in_features (int, optional): Input graph node dimensionality. Default to NODE_FEATURES.
            sigma (_type_, optional): Activation function. Default to torch.nn.ReLU().
            bias (bool, optional): Bias inclusion. Default to True.
            conv (_type_, optional): Convolutional Layer. Default to SAGEConv.
            pool (_type_, optional): Pooling Layer. Default to GraphMultisetTransformer.
        """
        super(TestNet, self).__init__()
        self.dim = dim
        self.node_features_list = features_list
        self.in_features = len(features_list)
        self.scaling_params = scaling_params
        self.sigma = sigma
        self.conv = conv
        self.num_conv_layers = N_conv
        self.num_linear_layers = N_linear
        self.adj_conv = adj_conv        
        # Building blocks of the GNN
        self.input_layer = Linear(self.in_features, self.dim, bias=bias)
        self.linear_block = torch.nn.ModuleList([Linear(self.dim, self.dim, bias=bias) for _ in range(self.num_linear_layers)])
        self.conv_block = torch.nn.ModuleList([conv(self.dim, self.dim, bias=bias) for _ in range(self.num_conv_layers)])
        if self.adj_conv:
            self.adj_block = torch.nn.ModuleList([Linear(self.dim, self.dim, bias=bias) for _ in range(self.num_conv_layers)])
        self.lin_a = Linear(self.dim, self.dim, bias=bias)        
        self.pma = PMA(channels = dim, 
                       num_heads = pool_heads, 
                       num_seeds = 1, 
                       bias = bias)
        self.lin_b = Linear(self.dim, 1, bias=bias)
        
    def forward(self, data):
        #------------#
        # NODE LEVEL #
        #------------#        
        out = self.sigma(self.input_layer(data.x))  # Input layer
        # Dense layers 
        for layer in range(self.num_linear_layers):
            out = self.sigma(self.linear_block[layer](out))
        # Convolutional layers
        for layer in range(self.num_conv_layers):
            if self.adj_conv:
                out = self.sigma(self.adj_block[layer](out))
            out = self.sigma(self.conv_block[layer](out, data.edge_index))
        #-----------------------#
        # GRAPH LEVEL (POOLING) #
        #-----------------------#
        out = self.lin_a(out)
        batch_x, mask = to_dense_batch(x=out, 
                                       batch=data.batch, # Tensor if batch, None if graph 
                                       fill_value=0.0, 
                                       max_num_nodes=500, # conservative value to avoid different predictions
                                       batch_size=None)  
        mask = (~mask).unsqueeze(1).to(dtype=out.dtype) * -1e9
        out = self.pma(x=batch_x, mask=mask)
        out = self.lin_b(out.squeeze(1))
        return out.view(-1)


class UQTestNet(torch.nn.Module):
    def __init__(self, 
                 features_list: list[str],
                 scaling_params: dict,                
                 dim: int,                  
                 N_linear: int=0,
                 N_conv: int=3,
                 adj_conv: bool=False,
                 sigma=torch.nn.ReLU(),
                 bias: bool=True,
                 conv=SAGEConv, 
                 pool_heads: int=4):
        """
        GNN with uncertainty quantification. output is a tuple of mean and variance.

        Args:
            dim (int, optional): Layer width.
            N_linear (int, optional): Number of dense. Default to 3.
            N_conv (int, optional): Number of convolutional layers. Default to 3.
            adj_conv (bool, optional): Whether include linear layer between each convolution. Default to True.
            in_features (int, optional): Input graph node dimensionality. Default to NODE_FEATURES.
            sigma (_type_, optional): Activation function. Default to torch.nn.ReLU().
            bias (bool, optional): Bias inclusion. Default to True.
            conv (_type_, optional): Convolutional Layer. Default to SAGEConv.
            pool (_type_, optional): Pooling Layer. Default to GraphMultisetTransformer.
        """
        super(UQTestNet, self).__init__()
        self.dim = dim
        self.node_features_list = features_list
        self.in_features = len(features_list)
        self.scaling_params = scaling_params
        self.sigma = sigma
        self.conv = conv
        self.num_conv_layers = N_conv
        self.num_linear_layers = N_linear
        self.adj_conv = adj_conv        
        # Building blocks of the GNN
        self.input_layer = Linear(self.in_features, self.dim, bias=bias)
        self.linear_block = torch.nn.ModuleList([Linear(self.dim, self.dim, bias=bias) for _ in range(self.num_linear_layers)])
        self.conv_block = torch.nn.ModuleList([conv(self.dim, self.dim, bias=bias) for _ in range(self.num_conv_layers)])
        if self.adj_conv:
            self.adj_block = torch.nn.ModuleList([Linear(self.dim, self.dim, bias=bias) for _ in range(self.num_conv_layers)])
        self.lin_a = Linear(self.dim, self.dim, bias=bias)        
        self.pma = PMA(channels = dim, 
                       num_heads = pool_heads, 
                       num_seeds = 1, 
                       bias = bias)
        self.lin_b = Linear(self.dim, 2, bias=bias)
        
    def forward(self, data):
        #------------#
        # NODE LEVEL #
        #------------#        
        out = self.sigma(self.input_layer(data.x))  # Input layer
        # Dense layers 
        for layer in range(self.num_linear_layers):
            out = self.sigma(self.linear_block[layer](out))
        # Convolutional layers
        for layer in range(self.num_conv_layers):
            if self.adj_conv:
                out = self.sigma(self.adj_block[layer](out))
            out = self.sigma(self.conv_block[layer](out, data.edge_index))
        #-----------------------#
        # GRAPH LEVEL (POOLING) #
        #-----------------------#
        out = self.lin_a(out)
        batch_x, mask = to_dense_batch(x=out, 
                                       batch=data.batch, # Tensor if batch, None if graph 
                                       fill_value=0.0, 
                                       max_num_nodes=500, # conservative value to avoid different predictions
                                       batch_size=None)  
        mask = (~mask).unsqueeze(1).to(dtype=out.dtype) * -1e9
        out = self.pma(x=batch_x, mask=mask)
        out = self.lin_b(out.squeeze(1))
        mu = out[:, 0]
        sigma = Softplus()(out[:, 1])
        normal = Normal(mu, sigma)
        #print(normal)
        return normal
