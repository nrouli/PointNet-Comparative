import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from utils import *
from metrics_tracker import MetricsTracker
import copy
from vn_pointnet_lite import VNPointNet

# load the seeds:
seeds = np.load('seeds.npy')

# select a seed:
SEED = seeds[0] 


torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PointNet Training')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--step', type=int, default=10, help='Logging interval')
    parser.add_argument('--rotate', action='store_true', help="Randomly rotate data during training")
    
    # Data parameters
    parser.add_argument('--num_points', type=int, default=1024, help='Number of points')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--data_type', type=str, default='clean', help='Data variant name')

    
    # Flags for which models to train
    parser.add_argument('--train_all', action='store_true', help='Train all models')
    parser.add_argument('--train_baseline', action='store_true', help='Train PointNet++')
    parser.add_argument('--train_geom', action='store_true', help='Train CGA-PointNet++')
    parser.add_argument('--train_pn', action='store_true', help= 'Train PointNet with PointCMLP version 0 (MLP)')
    parser.add_argument('--train_cpn', action='store_true', help='Train PointNet with PointCMLP version 1 (MLGP)')
    parser.add_argument('--train_vn', action='store_true', help= 'Train VN-PointNet')
   
    
    return parser.parse_args()



def train_model(model_name: str, data_type: str, model, train_data, validation_data, 
                rotate=False, epochs=30, step=1, batch_size=128, lr=1e-3, weight_decay=1e-4):
    
    Xtrain, Ytrain = train_data
    Xval, Yval = validation_data
    
    best_val_loss = float('inf')
    best_model_state = None  
    best_epoch = 0
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
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    n_batches = int(np.ceil(len(Xtrain) / batch_size))
    metrics = MetricsTracker(model_name + '_' + data_type)
    
    for i in range(epochs):
        model.train()
        
        for j in tqdm(range(n_batches), desc=f"Epoch {i+1}/{epochs}"):          
            Xbatch = Xtrain[j*batch_size:(j+1)*batch_size]
            Ybatch = Ytrain[j*batch_size:(j+1)*batch_size]
            if rotate:
                Rs = torch.stack([torch.tensor(uniform_random_rotation(), dtype=Xbatch.dtype) for _ in range(len(Xbatch))])
                Rs = Rs.cuda() if torch.cuda.is_available() else Rs
                Xbatch = torch.bmm(Xbatch, Rs)
            
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
                best_epoch = i
                # Save best model state
                best_model_state = {
                    'epoch': i + 1,
                    'state_dict': copy.deepcopy(model.state_dict()),
                    'optimizer': optimizer.state_dict().copy(),
                    'val_loss': val_loss,
                    'val_acc': val_acc
                }
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f'Early stopping at epoch {i}')
                    break
        
    
    metrics.save()
   
    # Load best model state before saving
    if best_model_state is not None:
        model.load_state_dict(best_model_state['state_dict'])
        print(f"\nLoaded best model from epoch {best_epoch} (val_loss: {best_val_loss:.4f})")
    
    # Save the best model
    full_name = model_name + '_' + data_type
    save_checkpoint(
        save_dir='pretrained_models',
        state={
            'model': model,
            'name': full_name,
            'epoch': best_model_state['epoch'] if best_model_state else i + 1,
            'state_dict': model.state_dict(),
            'optimizer': best_model_state['optimizer'] if best_model_state else optimizer.state_dict(),
            'best_val_loss': best_val_loss,
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
    
    
    
    # PointNet mlp
    if args.train_pn or args.train_all:
        model_name = "PointNet"
        
        model = build_point_net_mlp(output_dim=output_dim, 
                                    hidden_layer_sizes=[64, 64, 128, 64],
                                    activation=nn.functional.gelu,
                                    bias=True)
        metrics = train_model(
            model_name=model_name, 
            data_type=data_type, 
            model=model, 
            train_data=(Xtrain, Ytrain), 
            validation_data=(Xval, Yval),
            rotate=args.rotate,
            epochs=args.epochs, 
            step=args.step, 
            batch_size=args.batch_size,
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        all_metrics['PointNet'] = metrics
        plot_metrics(model_name + '_' + data_type, metrics)
    
    
    if args.train_cpn or args.train_all:
        model_name = "CGPointNet"
        model = build_point_net_mlgp(output_dim=output_dim,
                                     hidden_layer_sizes=[64, 62, 126, 64],
                                     activation=identity, 
                                     bias=False)
        metrics = train_model(
            model_name=model_name, 
            data_type=data_type, 
            model=model, 
            train_data=(Xtrain, Ytrain), 
            validation_data=(Xval, Yval),
            rotate=args.rotate,
            epochs=args.epochs, 
            step=args.step, 
            batch_size=args.batch_size,
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        all_metrics['CGPointNet'] = metrics
        plot_metrics(model_name + '_' + data_type, metrics)
        
        
    # PointNet++
    if args.train_baseline or args.train_all:
        model_name = "PointNet++"
        model = build_point_net_pp(output_dim=output_dim)
        metrics = train_model(
            model_name=model_name, 
            data_type=data_type, 
            model=model, 
            train_data=(Xtrain, Ytrain), 
            validation_data=(Xval, Yval),
            rotate=args.rotate,
            epochs=args.epochs, 
            step=args.step, 
            batch_size=args.batch_size,
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        all_metrics['PointNet++'] = metrics
        plot_metrics(model_name + '_' + data_type, metrics)
      

    # CGA-PointNet++
    if args.train_geom or args.train_all:
        model_name = "CGAPointNet++"
        model = build_cgapoint_net_pp(output_dim=output_dim)
        metrics = train_model(
            model_name=model_name, 
            data_type=data_type, 
            model=model, 
            train_data=(Xtrain, Ytrain), 
            validation_data=(Xval, Yval),
            rotate=args.rotate,
            epochs=args.epochs, 
            step=args.step, 
            batch_size=args.batch_size,
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        all_metrics['CGAPointNet++'] = metrics
        plot_metrics(model_name + '_' + data_type, metrics)
    
    if args.train_vn or args.train_all:
        model_name = "VN-PointNet"
        model = build_vn_point_net(output_dim=output_dim, base_channels=8, n_knn=20)
        metrics = train_model(
            model_name=model_name, 
            data_type=data_type, 
            model=model, 
            train_data=(Xtrain, Ytrain), 
            validation_data=(Xval, Yval),
            rotate=args.rotate,
            epochs=args.epochs, 
            step=args.step, 
            batch_size=args.batch_size,
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        all_metrics['VNPointNet'] = metrics
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
