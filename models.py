import torch
import torch.nn as nn
from torch_geometric.nn import fps, knn

import numpy as np

eps = 1e-8

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
        
        # create hidden layers:
        hidden_layers = []

        if version == 0:
            # for vanilla MLP:
            M1 = np.prod(input_shape)
            for M2 in hidden_layer_sizes:
                layer = nn.Linear(M1, M2, bias=bias)
                hidden_layers.append(layer)
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
                hidden_layers.append(layer)
                M1 = M2 + 2

            self.hidden_layers = nn.ModuleList(hidden_layers)

            #  the output layer:
            try:
                self.out_layer = nn.Linear(M2 + 2, output_dim, bias=bias)   
            except UnboundLocalError:
                self.out_layer = nn.Linear(M1, output_dim, bias=bias)   

            self.forward = self.forward_1
                        
        
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
    
    

class SetAbstraction(nn.Module):
    """
    Hierarchical Set Abstraction layer for the vanilla PointNet++ baseline.

    Downsamples point clouds via FPS, groups local neighborhoods with KNN,
    and processes grouped points (local coordinates concatenated with any
    existing features) through a shared Conv2d MLP with BatchNorm and ReLU.
    Local descriptors are aggregated per centroid using Log-Sum-Exp pooling.
    """
   
    
    def __init__(self, n_centroids, n_neighbors, in_dim, hidden_layer_size=[]):
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
        
        layers = []
        for out_dim in hidden_layer_size:
            layers.append(nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim
            
        self.mlp = nn.Sequential(*layers)
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
    
    def lse_pool(self, x, dim, tau=10.0):
        """
        Log-Sum-Exp pooling along dimension `dim`.
        x: tensor of shape (..., K, ...)
        """
        # subtract max for numerical stability
        x_max, _ = x.max(dim=dim, keepdim=True)
        lse = x_max + (1.0 / tau) * torch.log(
            torch.sum(torch.exp(tau * (x - x_max)), dim=dim, keepdim=True)
        )
        return lse

        
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


        grouped_features = grouped_features.permute(0, 3, 1, 2).contiguous()
        grouped_features = self.mlp(grouped_features)
    
        x = self.lse_pool(grouped_features, dim=3, tau=10).squeeze(3)
        x = x.permute(0, 2, 1).contiguous()   
        return centroids, x


class PointNetPP(nn.Module):
    """
    Vanilla PointNet++ classifier for 3D point cloud classification.

    Args:
        out_dim: Number of output classes.
    """
    
    def __init__(self, out_dim=10):
        super().__init__()
        
        # Hierarchical set abstraction layers
        self.sa1 = SetAbstraction(
            n_centroids=256, n_neighbors=16,
            in_dim=3, hidden_layer_size=[512, 256, 128])
        
        self.sa2 = SetAbstraction(
            n_centroids=64, n_neighbors=16,
            in_dim=128 + 3, hidden_layer_size=[128, 64])
        
        self.sa3 = SetAbstraction(
            n_centroids=1, n_neighbors=32 ,
            in_dim=64 + 3, hidden_layer_size=[64, 64])
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(64, 32, bias=False),  # bias is set to false here because batch normalization follows
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16, bias=False),  # bias is set to false here because batch normalization follows
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, out_dim)
        )
    
    def forward(self, xyz):
        # xyz: (B, N, 3)
        xyz1, feat1 = self.sa1(xyz, None)
        xyz2, feat2 = self.sa2(xyz1, feat1)
        _, feat3 = self.sa3(xyz2, feat2)
       
        x = feat3.squeeze(1) 
        return self.classifier(x)
    
    
    
class CGASetAbstraction(nn.Module):
    """
    Set Abstraction layer using a hybrid geometric-feature architecture.
    
    Downsamples point clouds via FPS, groups local neighborhoods with KNN,
    then processes geometry through a CGA-based PointCMLP (version=1) and
    per-point features through a standard MLP. Outputs are concatenated to
    produce a combined local descriptor per centroid.

    Args:
        n_centroids: Number of points to sample via FPS.
        n_neighbors: Number of nearest neighbors per centroid (K in KNN).
        feat_dim: Dimensionality of input per-point features. Set to 0
            for the first layer where no features exist yet.
        geo_hidden: Hidden layer sizes for the geometric PointCMLP.
            The last element becomes the geometric output dimension.
        feat_hidden: Hidden layer sizes for the feature MLP. Defaults
            to geo_hidden if not provided.
        activation: Activation function for the geometric MLP. The
            feature MLP always uses ReLU.
    """
    
    def __init__(
        self,
        n_centroids: int,
        n_neighbors: int,
        feat_dim: int,
        geo_hidden: list,
        feat_hidden: list = None,
        activation=lambda x: x,
    ):
        
        """
        Args:
            n_centroids: Number of points to sample via FPS.
            n_neighbors: Number of nearest neighbors per centroid (K in KNN).
            feat_dim: Feature channel dimension
            hidden_layer_size: List of output channel sizes for the shared MLP.
                The last element becomes the layer's output feature dimension.
        """
        super().__init__()
        self.n_centroids = n_centroids
        self.n_neighbors = n_neighbors
        self.feat_dim = feat_dim

        self.geo_mlp = PointCMLP(
            input_shape=(n_neighbors, 3),
            output_dim=geo_hidden[-1],
            hidden_layer_sizes=geo_hidden[:-1] if len(geo_hidden) > 1 else [],
            activation=activation,
            bias=False,
            version=1,
        )
        self.geo_out_dim = geo_hidden[-1]

        if feat_dim > 0:
            feat_hidden = feat_hidden or geo_hidden
            self.feat_mlp = PointCMLP(
                input_shape=(n_neighbors, feat_dim),
                output_dim=feat_hidden[-1],
                hidden_layer_sizes=feat_hidden[:-1] if len(feat_hidden) > 1 else [],
                activation=nn.functional.relu,
                bias=True,
                version=0,
            )
            self.feat_out_dim = feat_hidden[-1]
        else:
            self.feat_mlp = None
            self.feat_out_dim = 0

        self.out_dim = self.geo_out_dim + self.feat_out_dim

    def gather_points(self, points, idx):
        B = points.shape[0]
        if idx.dim() == 2:
            batch_idx = torch.arange(B, device=idx.device).view(-1, 1).expand_as(idx)
            return points[batch_idx, idx]
        elif idx.dim() == 3:
            B, M, K = idx.shape
            batch_idx = torch.arange(B, device=idx.device).view(B, 1, 1).expand(B, M, K)
            return points[batch_idx, idx]
        else:
            raise ValueError(f"idx must have dim 2 or 3, got {idx.dim()}")
    
    def knn_wrapper(self, xyz, centroids, k):
        B, N, _ = xyz.shape
        _, M, _ = centroids.shape
        xyz_flat = xyz.reshape(-1, 3)
        centroids_flat = centroids.reshape(-1, 3)
        batch_xyz = torch.arange(B, device=xyz.device).repeat_interleave(N)
        batch_centroids = torch.arange(B, device=xyz.device).repeat_interleave(M)
        edge_index = knn(
            x=xyz_flat,
            y=centroids_flat,
            k=k,
            batch_x=batch_xyz,
            batch_y=batch_centroids,
        )
        nn_flat = edge_index[1].view(B, M, k)
        idx = nn_flat % N
        return idx

    def forward(self, xyz, features):
        B, N, _ = xyz.shape
        xyz_flat = xyz.reshape(-1, 3)
        batch = torch.arange(B, device=xyz.device).repeat_interleave(N)
        idx_flat = fps(xyz_flat, batch, ratio=self.n_centroids / N)
        idx = idx_flat.reshape(B, -1) % N
        centroids = self.gather_points(xyz, idx)
        M = centroids.shape[1]

        with torch.no_grad():
            group_idx = self.knn_wrapper(xyz, centroids, self.n_neighbors)

        grouped_xyz = self.gather_points(xyz, group_idx)
        grouped_xyz = grouped_xyz - centroids.unsqueeze(2)

        x_geo = grouped_xyz.view(B * M, self.n_neighbors, 3)
        x_geo = self.geo_mlp(x_geo)
        x_geo = x_geo.view(B, M, -1)

        if features is not None and self.feat_mlp is not None:
            grouped_feat = self.gather_points(features, group_idx)
            x_feat = grouped_feat.view(B * M, self.n_neighbors, -1)
            x_feat = self.feat_mlp(x_feat)
            x_feat = x_feat.view(B, M, -1)
            x = torch.cat([x_geo, x_feat], dim=-1)
        else:
            x = x_geo

        return centroids, x


class CGAPointNetPP(nn.Module):
    """
    CGA-augmented PointNet++ classifier for 3D point cloud classification.
    """
    
    
    def __init__(self, out_dim=10, activation=lambda x: x):
        """
        Args:
        out_dim: Number of output classes.
        activation: Activation function for the CGA geometric MLPs.
        """
        
        super().__init__()
        self.sa1 = CGASetAbstraction(
            n_centroids=256,
            n_neighbors=64,
            feat_dim=0,
            geo_hidden=[32],
            activation=activation,
        )
        
        self.sa2 = CGASetAbstraction(
            n_centroids=64,
            n_neighbors=32,
            feat_dim=self.sa1.out_dim,
            geo_hidden=[32],
            feat_hidden=[128, 64],
            activation=activation,
        )

        self.final_geo = PointCMLP(
            input_shape=(self.sa2.n_centroids, 3),
            output_dim=128,
            hidden_layer_sizes=[16],
            activation=activation,
            bias=False,
            version=1,
        )
        
        self.final_feat = nn.Sequential(
            nn.Linear(self.sa2.out_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256))

        self.classifier = nn.Linear(128 + 256, out_dim, bias=True)

    def forward(self, xyz):
        xyz1, feat1 = self.sa1(xyz, None)
        xyz2, feat2 = self.sa2(xyz1, feat1)

        x_geo = self.final_geo(xyz2)
        x_feat = feat2.max(dim=1)[0]  
        x_feat = self.final_feat(x_feat)
        
        x = torch.cat([x_geo, x_feat], dim=-1)
        
        return self.classifier(x)
        
