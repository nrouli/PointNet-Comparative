import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6


def knn(x, k):
    """Find k nearest neighbors for each point."""
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature_cross(x, k=20, idx=None):
    """Compute graph features with cross products for rotation equivariance."""
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)
    
    if idx is None:
        idx = knn(x, k=k)
    
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3)
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)
    cross = torch.cross(feature, x, dim=-1)
    
    feature = torch.cat((feature - x, x, cross), dim=3).permute(0, 3, 4, 1, 2).contiguous()
    return feature


class VNLinear(nn.Module):
    """Equivariant linear layer for vector features."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        # x: [B, N_feat, 3, N_samples, ...] 
        return self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)


class VNBatchNorm(nn.Module):
    """Equivariant batch normalization."""
    def __init__(self, num_features, dim):
        super().__init__()
        self.dim = dim
        if dim == 3 or dim == 4:
            self.bn = nn.BatchNorm1d(num_features)
        elif dim == 5:
            self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x):
        norm = torch.norm(x, dim=2) + EPS
        norm_bn = self.bn(norm)
        norm = norm.unsqueeze(2)
        norm_bn = norm_bn.unsqueeze(2)
        return x / norm * norm_bn


class VNLeakyReLU(nn.Module):
    """Equivariant LeakyReLU activation."""
    def __init__(self, in_channels, share_nonlinearity=False, negative_slope=0.2):
        super().__init__()
        out_ch = 1 if share_nonlinearity else in_channels
        self.map_to_dir = nn.Linear(in_channels, out_ch, bias=False)
        self.negative_slope = negative_slope

    def forward(self, x):
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (x * d).sum(2, keepdim=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d * d).sum(2, keepdim=True)
        x_out = self.negative_slope * x + (1 - self.negative_slope) * (
            mask * x + (1 - mask) * (x - (dotprod / (d_norm_sq + EPS)) * d)
        )
        return x_out


class VNLinearLeakyReLU(nn.Module):
    """Combined linear + batchnorm + leaky relu."""
    def __init__(self, in_channels, out_channels, dim=5, share_nonlinearity=False, negative_slope=0.0):
        super().__init__()
        self.dim = dim
        self.negative_slope = negative_slope

        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)
        self.batchnorm = VNBatchNorm(out_channels, dim=dim)

        out_ch = 1 if share_nonlinearity else out_channels
        self.map_to_dir = nn.Linear(in_channels, out_ch, bias=False)

    def forward(self, x):
        p = self.map_to_feat(x.transpose(1, -1)).transpose(1, -1)
        p = self.batchnorm(p)
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (p * d).sum(2, keepdims=True)
        mask = (dotprod >= 0).float()
        d_norm_sq = (d * d).sum(2, keepdims=True)
        x_out = self.negative_slope * p + (1 - self.negative_slope) * (
            mask * p + (1 - mask) * (p - (dotprod / (d_norm_sq + EPS)) * d)
        )
        return x_out


class VNMaxPool(nn.Module):
    """Equivariant max pooling."""
    def __init__(self, in_channels):
        super().__init__()
        self.map_to_dir = nn.Linear(in_channels, in_channels, bias=False)

    def forward(self, x):
        d = self.map_to_dir(x.transpose(1, -1)).transpose(1, -1)
        dotprod = (x * d).sum(2, keepdims=True)
        idx = dotprod.max(dim=-1, keepdim=False)[1]
        index_tuple = torch.meshgrid([torch.arange(j) for j in x.size()[:-1]], indexing='ij') + (idx,)
        x_max = x[index_tuple]
        return x_max


def mean_pool(x, dim=-1, keepdim=False):
    return x.mean(dim=dim, keepdim=keepdim)


class VNStdFeature(nn.Module):
    """Extract invariant features from equivariant representations."""
    def __init__(self, in_channels, dim=4, normalize_frame=False, share_nonlinearity=False, negative_slope=0.2):
        super().__init__()
        self.dim = dim
        self.normalize_frame = normalize_frame

        self.vn1 = VNLinearLeakyReLU(in_channels, in_channels // 2, dim=dim, 
                                      share_nonlinearity=share_nonlinearity, negative_slope=negative_slope)
        self.vn2 = VNLinearLeakyReLU(in_channels // 2, in_channels // 4, dim=dim,
                                      share_nonlinearity=share_nonlinearity, negative_slope=negative_slope)
        
        out_vecs = 2 if normalize_frame else 3
        self.vn_lin = nn.Linear(in_channels // 4, out_vecs, bias=False)

    def forward(self, x):
        z0 = self.vn1(x)
        z0 = self.vn2(z0)
        z0 = self.vn_lin(z0.transpose(1, -1)).transpose(1, -1)

        if self.normalize_frame:
            v1 = z0[:, 0, :]
            v1_norm = torch.sqrt((v1 * v1).sum(1, keepdims=True))
            u1 = v1 / (v1_norm + EPS)
            v2 = z0[:, 1, :]
            v2 = v2 - (v2 * u1).sum(1, keepdims=True) * u1
            v2_norm = torch.sqrt((v2 * v2).sum(1, keepdims=True))
            u2 = v2 / (v2_norm + EPS)
            u3 = torch.cross(u1, u2)
            z0 = torch.stack([u1, u2, u3], dim=1).transpose(1, 2)
        else:
            z0 = z0.transpose(1, 2)

        if self.dim == 4:
            x_std = torch.einsum('bijm,bjkm->bikm', x, z0)
        elif self.dim == 3:
            x_std = torch.einsum('bij,bjk->bik', x, z0)
        elif self.dim == 5:
            x_std = torch.einsum('bijmn,bjkmn->bikmn', x, z0)

        return x_std, z0


class VNPointNetEncoder(nn.Module):
    """
    VN-PointNet encoder with configurable channel widths.
    """
    def __init__(self, n_knn=20, pooling='mean', base_channels=16):
        super().__init__()
        self.n_knn = n_knn
        
       
        c1 = base_channels     
        c2 = base_channels * 2   
        c3 = base_channels * 4   

        # Input conv: processes graph features (3 types × 3 = 9 vector channels -> c1)
        self.conv_pos = VNLinearLeakyReLU(3, c1, dim=5, negative_slope=0.0)
        
        # Point-wise convolutions
        self.conv1 = VNLinearLeakyReLU(c1, c1, dim=4, negative_slope=0.0)
        self.conv2 = VNLinearLeakyReLU(c1 * 2, c2, dim=4, negative_slope=0.0)  # after concat with global
        
        self.conv3 = VNLinear(c2, c3)
        self.bn3 = VNBatchNorm(c3, dim=4)
        
        # For extracting invariant features
        self.std_feature = VNStdFeature(c3 * 2, dim=4, normalize_frame=False, negative_slope=0.0)
        
        # Pooling
        if pooling == 'max':
            self.pool = VNMaxPool(c1)
        else:
            self.pool = mean_pool
            
        self.out_channels = c3 * 2 * 3  # c3*2 vector channels × 3 (from std_feature)

    def forward(self, x):
        """
        x: (B, 3, N) - point cloud in channels-first format
        returns: (B, out_channels) - global features
        """
        B, D, N = x.size()
        
        x = x.unsqueeze(1)  # (B, 1, 3, N)
        feat = get_graph_feature_cross(x, k=self.n_knn)  # (B, 3, 3, N, k)
        x = self.conv_pos(feat)  # (B, c1, 3, N, k)
        x = self.pool(x)  # (B, c1, 3, N)

        x = self.conv1(x)  # (B, c1, 3, N)
        
        # Global feature concatenation (lightweight version - no separate transform network)
        x_global = x.mean(dim=-1, keepdim=True).expand(-1, -1, -1, N)
        x = torch.cat((x, x_global), 1)  # (B, c1*2, 3, N)
        
        x = self.conv2(x)  # (B, c2, 3, N)
        x = self.bn3(self.conv3(x))  # (B, c3, 3, N)

        # Concat with mean for std_feature
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)  # (B, c3*2, 3, N)
        
        x, trans = self.std_feature(x)  # x: (B, c3*2, 3, N)
        x = x.view(B, -1, N)  # (B, c3*2*3, N)

        x = torch.max(x, -1)[0]  # (B, c3*2*3)
        
        return x


class VNPointNet(nn.Module):
    """
    Lightweight VN-PointNet classifier matching PointNet++/CGA-PointNet++ scale.
    
    Args:
        num_classes: Number of output classes
        base_channels: Base channel width
        n_knn: Number of neighbors for graph construction
        pooling: 'mean' or 'max' for VN pooling
    """
    def __init__(self, num_classes=10, base_channels=16, n_knn=20, pooling='mean'):
        super().__init__()
        
        self.encoder = VNPointNetEncoder(
            n_knn=n_knn, 
            pooling=pooling, 
            base_channels=base_channels
        )
        
        enc_out = self.encoder.out_channels  # base_channels * 4 * 2 * 3 = 384 for base=16
        
        # Classifier matching your PointNet++ structure
        self.classifier = nn.Sequential(
            nn.Linear(enc_out, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        """
        x: (B, N, 3) or (B, 3, N) - handles both formats
        returns: (B, num_classes) - raw logits (use with CrossEntropyLoss)
        """
        # Handle both input formats
        if x.size(1) != 3 and x.size(2) == 3:
            x = x.transpose(1, 2)  # (B, N, 3) -> (B, 3, N)
        
        features = self.encoder(x)
        return self.classifier(features)


def get_vn_model_lite(num_class=10, base_channels=16, n_knn=20, pooling='mean'):
    """
    Factory function matching the interface expected by train.py
    """
    return VNPointNet(
        num_classes=num_class,
        base_channels=base_channels,
        n_knn=n_knn,
        pooling=pooling
    )


# For comparison with original
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test and compare parameter counts
    print("Parameter comparison:")
    print("-" * 50)
    
    for base_ch in [8, 12, 16, 20, 24]:
        model = VNPointNet(num_classes=10, base_channels=base_ch)
        print(f"VN-PointNet Lite (base={base_ch:2d}): {count_parameters(model):,} params")
    
    print("-" * 50)
    print("\nTest forward pass:")
    model = VNPointNet(num_classes=10, base_channels=16)
    x = torch.randn(4, 1024, 3)  # (B, N, 3) format
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
