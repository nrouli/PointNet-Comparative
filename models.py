import torch
import torch.nn as nn
from torch_geometric.nn import fps, knn
import torch.nn.init as init
import numpy as np


# The original authors of class PointCMLP are Pavlo Melnyk, Michael Felsberg, Marten Wadenback
class PointCMLP(nn.Module):
    """A single class to create all types of models in the experiments."""
    
    def __init__(self, input_shape, output_dim, hidden_layer_sizes=[], activation=lambda x: x, bias=False, version=1):
        """
        Args:
            input_shape:        a list/tuple of 2 integers; the size of one input sample, i.e., (n_rows, n_columns);
                                the model input is, however, expected to be a 3D array of shape (n_samples, n_rows, n_columns);
            hidden_layer_sizes: a list of integers containing the number of hidden units in each layer;
            activation:         activation function, e.g., nn.functional.relu;
            output_dim:         integer; the number of output units.
            version:            either 0 or 1:
                                0 to create a vanilla MLP (the input will be vectorized in the forward function);
                                1 to create the proposed MLGP or the baseline MLHP.
                                For the former, the embedding of the input is row-wise.
                                In order to create the latter, one needs to vectorize each sample in the input, i.e.,
                                reshape the input to (n_samples, 1, n_rows*n_columns).
        """
        super().__init__()

        self.input_shape = input_shape
        self.f = activation
        self.version = version
        # create hidden layers:
        hidden_layers = []

        if version == 0:
            # for vanilla MLP:
            M1 = np.prod(input_shape)
            for M2 in hidden_layer_sizes:
                layer = nn.Linear(M1, M2, bias=bias)
                hidden_layers.append(nn.Sequential(layer, nn.LayerNorm(M2)))
                M1 = M2

            self.hidden_layers = nn.ModuleList(hidden_layers)

            # the output layer:
            try:
                self.out_layer = nn.Linear(M2, output_dim, bias=bias)   
            except UnboundLocalError:
                self.out_layer = nn.Linear(M1, output_dim, bias=bias)                   

            self.forward = self.forward_0


        elif version == 1:
            # for the proposed MLGP and the baseline MLHP

            # for input_shape = (k, 3), e.g., 3D shape coordinates,
            # each row of the input sample (R^3) is embedded in R^5 (ME^3);
            # the resulting (k x 5)-array is vectorized row-wise and fed
            # to the first layer;
            # each subsequent hidden layer output R^n is first embedded in R^(n+2)
            # and then fed to the next layer

            M1 = input_shape[0] * (input_shape[1] + 2)
            for M2 in hidden_layer_sizes:
                layer = nn.Linear(M1, M2, bias=bias)
                hidden_layers.append(nn.Sequential(layer, nn.LayerNorm(M2)))
                M1 = M2 + 2

            self.hidden_layers = nn.ModuleList(hidden_layers)

            #  the output layer:
            try:
                self.out_layer = nn.Linear(M2 + 2, output_dim, bias=bias)   
            except UnboundLocalError:
                self.out_layer = nn.Linear(M1, output_dim, bias=bias)   

            self.forward = self.forward_1
        
        self._initialize_weights()
            
    # initialization did not exist in the original implementation    
    def _initialize_weights(self):
        if self.version == 0:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    init.kaiming_normal_(m.weight)
            
        else:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    init.orthogonal_(m.weight, gain=1.0)
                    if m.bias is not None:
                        init.constant_(m.bias, 0)
               
                        
        
    def forward_0(self, x):
        # for the vanilla MLP
        # vectorize each sample:
        x = x.view(-1, x.shape[1]*x.shape[2]) 

        for layer in self.hidden_layers:
            x = self.f(layer(x))
        x = self.out_layer(x)

        return x


    def forward_1(self, x): 
        # for the proposed MLGP and the baseline MLHP   

        embed_term_1 = -torch.ones(len(x), x.shape[1], 1).float()
        embed_term_2 = -torch.sum(x**2, axis=2) / 2 

        if torch.cuda.is_available():
            embed_term_1 = embed_term_1.cuda()

        x = torch.cat((x, embed_term_1, embed_term_2.view(-1, x.shape[1], 1)), dim=2)
        x = x.view(-1, x.shape[1]*x.shape[2]) 

        for layer in self.hidden_layers:
            x = self.f(layer(x))

            embed_term_1 = -torch.ones(len(x), 1).float()
            embed_term_2 = -torch.sum(x**2, axis=1).view(-1, 1) / 2

            if torch.cuda.is_available():
                embed_term_1 = embed_term_1.cuda()

            x = torch.cat((x, embed_term_1, embed_term_2), dim=1)

        x = self.out_layer(x)

        return x
    
    
class PointNet(nn.Module):
    def __init__(self, out_dim, hidden_layer_sizes=[], activation=lambda x:x, bias=False, version=0):
        super().__init__()
        self.point_mlp = PointCMLP(input_shape=(1,3),
                             output_dim=hidden_layer_sizes[-1],
                             hidden_layer_sizes=hidden_layer_sizes[:-1],
                             activation=activation,
                             bias=bias,
                             version=version)
        
        self.version = version
        self.fc = nn.Linear(hidden_layer_sizes[-1], out_dim)
        self._initialize_weights()
        
    
    def _initialize_weights(self):
        
        if self.version == 0:
            for m in self.fc.modules():
                if isinstance(m, nn.Linear):
                    init.kaiming_normal_(m.weight)
            
        else:
            for m in self.fc.modules():
                if isinstance(m, nn.Linear):
                    init.orthogonal_(m.weight, gain=1.0)
        
        if self.fc.bias is not None:
            init.constant_(self.fc.bias, 0)
            

    def forward(self, x):
        B, N, _ = x.shape
        
        x = x.reshape(B*N, 1, 3)
        x = self.point_mlp(x)
        
        x = x.reshape(B, N,-1)
        
        x = x.max(dim=1)[0]
        x = self.fc(x)
        return x
    
    

class SetAbstraction(nn.Module):
    """
    Hierarchical Set Abstraction layer for the vanilla PointNet++ baseline.

    Downsamples point clouds via FPS, groups local neighborhoods with KNN,
    and processes grouped points (local coordinates concatenated with any
    existing features) through a shared MLP with BatchNorm and ReLU.
    Local descriptors are aggregated per centroid using max pooling.
    """
   
    
    def __init__(self, 
                 n_centroids, 
                 n_neighbors, 
                 in_dim, 
                 hidden_layer_size=[], 
                 activation= lambda x:x, 
                 bias=True, 
                 version=0):
        """
        Args:
            n_centroids: Number of points to sample via FPS.
            n_neighbors: Number of nearest neighbors per centroid (K in KNN).
            in_dim: Input channel dimension.
            hidden_layer_size: List of output channel sizes for the shared MLP.
                The last element becomes the layer's output feature dimension.
        """
        
        super().__init__()
        self.n_centroids = n_centroids
        self.n_neighbors = n_neighbors

            
        self.mlp = PointCMLP(input_shape=(1, in_dim),
                             output_dim=hidden_layer_size[-1],
                             hidden_layer_sizes=hidden_layer_size[:-1], 
                             activation=activation, 
                             bias=bias, 
                             version=version)
        self.out_dim = hidden_layer_size[-1]
        
        
    def gather_points(self, points, idx):
        """
        Args:
            points: (B, N, C) - point cloud or features
            idx: (B, M) or (B, M, K) - indices to gather
        Returns:
            gathered: (B, M, C) or (B, M, K, C)
        """
        B = points.shape[0]
        
        if idx.dim() == 2:
            # Simple case: (B, M) indices -> (B, M, C) output
            batch_idx = torch.arange(B, device=idx.device).view(-1, 1).expand_as(idx)
            return points[batch_idx, idx]
        
        elif idx.dim() == 3:
            # Grouped case: (B, M, K) indices -> (B, M, K, C) output
            B, M, K = idx.shape
            batch_idx = torch.arange(B, device=idx.device).view(B, 1, 1).expand(B, M, K)
            return points[batch_idx, idx]
        
    
    def knn_wrapper(self, xyz, centroids, k):
        """
        knn wrapper function that calls knn from torch_geometric library
        """
        B, N, _ = xyz.shape
        _, M, _ = centroids.shape
        
        xyz_flat = xyz.reshape(-1, 3)
        centroids_flat = centroids.reshape(-1, 3)
        batch_xyz = torch.arange(B, device=xyz.device).repeat_interleave(N)
        batch_centroids = torch.arange(B, device=xyz.device).repeat_interleave(M)
        
        edge_index = knn(xyz_flat, centroids_flat, k, batch_xyz, batch_centroids)
        
        nn_flat = edge_index[1].view(B, M, k)
        idx = nn_flat - (torch.arange(B, device=xyz.device).view(B,1,1) * N)
        
        return idx

        
    def forward(self, xyz, features):
        B, N, _ = xyz.shape
        
        xyz_flat = xyz.reshape(-1, 3)
        batch = torch.arange(B, device=xyz.device).repeat_interleave(N)
        
        idx_flat = fps(xyz_flat, batch, ratio=self.n_centroids / N)
        idx = idx_flat.reshape(B, -1) % N
        
        centroids = self.gather_points(xyz, idx)
        
        # k-NN grouping
        with torch.no_grad():
            group_idx = self.knn_wrapper(xyz, centroids, self.n_neighbors)  # (B, n_centroids, n_neighbors)
            grouped_xyz = self.gather_points(xyz, group_idx)  # (B, n_centroids, n_neighbors, 3)
            
        # Normalize to local coordinates
        grouped_xyz = grouped_xyz - centroids.unsqueeze(2)
        
        # Concatenate with features if available
        if features is not None:
            grouped_features = self.gather_points(features, group_idx)
            grouped_features = torch.cat([grouped_xyz, grouped_features], dim=-1)
        else:
            grouped_features = grouped_xyz
            
        
        B, M, K, D = grouped_features.shape
        x = grouped_features.view(B * M * K, 1, D)     # each neighbor independently
        x = self.mlp(x)                                # (B*M*K, out_dim)
        x = x.view(B * M, K, -1)                       # (B*M, K, out_dim)
        x = x.max(dim=1)[0]                            # (B*M, out_dim)
        x = x.view(B, M, -1)                           # (B, M, out_dim)
        return centroids, x


class PointNetPP(nn.Module):
    """
    PointNet++ classifier for 3D point cloud classification.
    Supports both vanilla and CGA-augmented modes.

    Args:
        out_dim: Number of output classes.
        cga: If True, use CGA geometric MLPs (version=1, no bias, hidden=62).
        activation: Activation function for the MLPs.
    """
    
    def __init__(self, out_dim=10, cga=False, activation=lambda x: x):
        super().__init__()
        
        if cga:
            hidden, bias, version = [62, 64], False, 1
            sa_activation = activation
        else:
            hidden, bias, version = [64, 64], True, 0
            sa_activation = activation
        
        # Hierarchical set abstraction layers
        self.sa1 = SetAbstraction(
            n_centroids=256, n_neighbors=32,
            in_dim=3, hidden_layer_size=hidden,
            activation=sa_activation if cga else nn.functional.gelu,
            bias=bias if cga else True,
            version=version if cga else 0)
        
        self.sa2 = SetAbstraction(
            n_centroids=64, n_neighbors=32,
            in_dim=64 + 3, hidden_layer_size=hidden if cga else [64, 64],
            activation=sa_activation if cga else nn.functional.gelu,
            bias=bias if cga else True,
            version=version if cga else 0)
        
        self.sa3 = SetAbstraction(
            n_centroids=1, n_neighbors=64,
            in_dim=64 + 3, hidden_layer_size=hidden if cga else [64, 64],
            activation=sa_activation if cga else nn.functional.gelu,
            bias=bias if cga else True,
            version=version if cga else 0)
        
        self.classifier = PointCMLP(input_shape=(1, self.sa3.out_dim), output_dim=out_dim, hidden_layer_sizes=[32], activation=activation, bias=bias, version=version)
        
    #     self._initialize_weights()
        
    # def _initialize_weights(self):
    #     for m in self.classifier.modules():
    #         if isinstance(m, nn.Linear):
    #             init.kaiming_normal_(m.weight)
    #             if m.bias is not None:
    #                 init.constant_(m.bias, 0)
    
    def forward(self, xyz):
        xyz1, feat1 = self.sa1(xyz, None)
        xyz2, feat2 = self.sa2(xyz1, feat1)
        _, feat3 = self.sa3(xyz2, feat2)
        # x = feat3.squeeze(1)
        return self.classifier(feat3)