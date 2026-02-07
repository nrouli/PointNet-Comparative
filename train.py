import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from utils import *
from metrics_tracker import MetricsTracker

# load the seeds:
seeds = np.load('seeds.npy')

# select a seed:
SEED = seeds[0] 


torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



# Import the VN model
from vn_pointnet_lite import VNPointNet


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet Training')
    
    # Model selection
    parser.add_argument('--model', default='vn_pointnet_lite', 
                    choices=['pointnet_pp', 'cga_pointnet_pp', 'vn_pointnet_lite', 'vn_pointnet_orig', 'mlgp'],
                    help='Model to train')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--step', type=int, default=10, help='Logging interval')
    
    # Data parameters
    parser.add_argument('--num_points', type=int, default=1024, help='Number of points')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--data_type', type=str, default='clean', help='Data variant name')
    
    # VN-specific parameters
    parser.add_argument('--base_channels', type=int, default=12, 
                        help='Base channel width for VN-PointNet Lite (8=~15K, 16=~35K, 24=~70K params)')
    parser.add_argument('--n_knn', type=int, default=20, help='Number of KNN neighbors')
    parser.add_argument('--pooling', type=str, default='mean', choices=['mean', 'max'],
                        help='Pooling method for VN layers')
    
    # Original VN model parameters (if using vn_pointnet_orig)
    parser.add_argument('--rot', type=str, default='aligned', choices=['aligned', 'z', 'so3'],
                        help='Rotation augmentation')
    parser.add_argument('--normal', action='store_true', default=False, 
                        help='Use normal information')
    
    # MLGP Parameters
    parser.add_argument('--mlgp_points', type=int, default=4, 
                    help='Number of points for MLGP input (data will be subsampled)')
    parser.add_argument('--mlgp_hidden', type=int, nargs='+', default=[20],
                    help='Hidden layer sizes for MLGP')
    
    # Flags for which models to train
    parser.add_argument('--train_baseline', action='store_true', help='Train PointNet++')
    parser.add_argument('--train_geom', action='store_true', help='Train CGA-PointNet++')
    parser.add_argument('--train_vn', action='store_true', help='Train VN-PointNet')
    parser.add_argument('--train_mlgp', action='store_true', help='Train MLGP')
    
    return parser.parse_args()


def build_model(args, output_dim):
    """Build the specified model."""
    
    if args.model == 'vn_pointnet_lite':
        model = VNPointNet(
            num_classes=output_dim,
            base_channels=args.base_channels,
            n_knn=args.n_knn,
            pooling=args.pooling
        )
        
    elif args.model == 'vn_pointnet_orig':
        # Use original VN model (requires vn_pointnet_cls.py)
        from vn_pointnet_cls import get_vn_model as OriginalVNModel
        
        # Create a wrapper that handles the output format
        class VNModelWrapper(nn.Module):
            def __init__(self, vn_model):
                super().__init__()
                self.model = vn_model
                
            def forward(self, x):
                # Handle input shape: (B, N, 3) -> (B, 3, N)
                if x.size(1) != 3 and x.size(2) == 3:
                    x = x.transpose(1, 2)
                    
                out, _ = self.model(x)  # Unpack (logits, trans_feat)
                
                return torch.exp(out)
                
        model = VNModelWrapper(OriginalVNModel(args, num_class=output_dim, normal_channel=False))
        
    elif args.model == 'pointnet_pp':
        model = build_point_net_pp(output_dim=output_dim, activation=nn.functional.relu, 
                                   dropout=0.2, version=0)
        
    elif args.model == 'cga_pointnet_pp':
        model = build_cgapoint_net_pp(output_dim=output_dim)
        
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    return model


def train_model(model_name: str, data_type: str, model, train_data, validation_data, 
                epochs, step, batch_size=128, lr=1e-3, weight_decay=1e-4):
    
    Xtrain, Ytrain = train_data
    Xval, Yval = validation_data
    
    best_val_loss = float('inf')
    patience = epochs - 2 * epochs // 4
    no_improve = 0
    
    torch.manual_seed(SEED)

    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")
    print(model)
    n_params = sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())])
    print(f'Total parameters: {n_params:,}')
    print()

    if torch.cuda.is_available():
        model = model.cuda()
        Xtrain, Ytrain = Xtrain.float().cuda(), Ytrain.cuda()
        Xval, Yval = Xval.float().cuda(), Yval.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    n_batches = int(np.ceil(len(Xtrain) / batch_size))
    metrics = MetricsTracker(model_name + '_' + data_type)
    
    for i in range(epochs):
        model.train()
        
        for j in tqdm(range(n_batches), desc=f"Epoch {i+1}/{epochs}"):          
            Xbatch = Xtrain[j*batch_size:(j+1)*batch_size]
            Ybatch = Ytrain[j*batch_size:(j+1)*batch_size]

            y_pred = model(Xbatch)
            loss = criterion(y_pred, Ybatch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if i % step == 0:
            with torch.no_grad():
                loss, acc = evaluate(model, Xtrain, Ytrain, criterion)
                val_loss, val_acc = evaluate(model, Xval, Yval, criterion)
                
            metrics.update(i, loss, val_loss, acc, val_acc)
            print(f'Epoch {i:3d} | Loss: {loss:.4f} | Val Loss: {val_loss:.4f} | '
                  f'Acc: {acc:.4f} | Val Acc: {val_acc:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'Early stopping at epoch {i}')
                break

    metrics.save()
   
    # Save the model
    full_name = model_name + '_' + data_type
    save_checkpoint(
        save_dir='pretrained_models',
        state={
            'model': model,
            'name': full_name,
            'epoch': i + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'seed': SEED,
        }
    )
    
    return metrics


def evaluate(model, X, Y, criterion, batch_size=256):
    model.eval()
    total_loss = 0
    correct = 0
    n_batches = (len(X) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for j in range(n_batches):
            Xbatch = X[j*batch_size:(j+1)*batch_size]
            Ybatch = Y[j*batch_size:(j+1)*batch_size]
            y_pred = model(Xbatch)
            total_loss += criterion(y_pred, Ybatch).item()
            correct += (y_pred.argmax(dim=1) == Ybatch).sum().item()
    
    return total_loss / n_batches, correct / len(X)


def plot_metrics(model_name, metrics):
    
    fig, ax = metrics.plot(metric='accuracy')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_title('Accuracy')
    fig.savefig(model_name + '_accuracy.pdf', bbox_inches='tight')
    
    fig, ax = metrics.plot(metric='loss')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_title('Loss')
    fig.savefig(model_name + '_loss.pdf', bbox_inches='tight')
    

def main():
    args = parse_args()
    
    # Load data
    (Xtrain, Ytrain), (Xval, Yval) = get_model_net_data(
        train_size=3991, 
        test_size=908, 
        n_points=1024,
        distortion=0.0,
        class_size=args.num_classes,
        force_reload=False
    )
    output_dim = len(set(Ytrain.numpy()))
    data_type = args.data_type
    
    all_metrics = {}
    
    
    # 1) PointNet++
    if args.train_baseline:
        model_name = "PointNet++"
        model = build_point_net_pp(output_dim=output_dim)
        metrics = train_model(
            model_name=model_name, 
            data_type=data_type, 
            model=model, 
            train_data=(Xtrain, Ytrain), 
            validation_data=(Xval, Yval), 
            epochs=args.epochs, 
            step=args.step, 
            batch_size=args.batch_size,
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        all_metrics['PointNet++'] = metrics
        plot_metrics(model_name + '_' + data_type, metrics)
      
        
    # 2) CGA-PointNet++
    if args.train_geom:
        model_name = "CGAPointNet++"
        model = build_cgapoint_net_pp(output_dim=output_dim)
        metrics = train_model(
            model_name=model_name, 
            data_type=data_type, 
            model=model, 
            train_data=(Xtrain, Ytrain), 
            validation_data=(Xval, Yval), 
            epochs=args.epochs, 
            step=args.step, 
            batch_size=args.batch_size,
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        all_metrics['CGAPointNet++'] = metrics
        plot_metrics(model_name + '_' + data_type, metrics)
        
        
    # 3) VN-PointNet (Lite version)
    if args.train_vn:
        model_name = "VNPointNet"
        model = VNPointNet(
            num_classes=output_dim,
            base_channels=args.base_channels,
            n_knn=args.n_knn,
            pooling=args.pooling
        )
        metrics = train_model(
            model_name=model_name, 
            data_type=data_type, 
            model=model, 
            train_data=(Xtrain, Ytrain), 
            validation_data=(Xval, Yval), 
            epochs=args.epochs, 
            step=args.step, 
            batch_size=args.batch_size,
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        all_metrics['VNPointNet_Lite'] = metrics
        plot_metrics(model_name + '_' + data_type, metrics)
        
        
    # 4) MLGP
    if args.train_mlgp:
        model_name = "MLGP"
        
        # MLGP needs smaller point clouds - subsample
        n_pts = 256
        Xtrain_mlgp = Xtrain[:, :n_pts, :]
        Xval_mlgp = Xval[:, :n_pts, :]
        
        model = build_mlgp(
            input_shape=(n_pts, 3),
            output_dim=output_dim,
            hidden_layer_sizes=[64, 64, 32],
            bias=False,
        )
        metrics = train_model(
            model_name=model_name, 
            data_type=data_type, 
            model=model, 
            train_data=(Xtrain_mlgp, Ytrain), 
            validation_data=(Xval_mlgp, Yval), 
            epochs=args.epochs, 
            step=args.step, 
            batch_size=args.batch_size,
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        all_metrics['MLGP'] = metrics
        plot_metrics(model_name + '_' + data_type, metrics)
    
    
    # Comparative plots if multiple models trained
    if len(all_metrics) > 1:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots()
        for name, m in all_metrics.items():
            fig, ax = m.plot('loss', include_validation=True, ax=ax)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_title('Comparative Loss')
        fig.savefig(f'Comparative_loss_{data_type}.pdf', dpi=300, bbox_inches='tight')
        
        fig, ax = plt.subplots()
        for name, m in all_metrics.items():
            fig, ax = m.plot('accuracy', include_validation=True, ax=ax)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_title('Comparative Accuracy')
        fig.savefig(f'Comparative_accuracy_{data_type}.pdf', bbox_inches='tight')



if __name__ == '__main__':
    main()
