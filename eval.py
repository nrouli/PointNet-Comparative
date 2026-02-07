import torch
import numpy as np
from utils import *
import matplotlib.pyplot as plt
from time import time


# MODEL_PATH = 'pretrained_models/VNPointNet_Lite_clean.tar'
MODEL_PATH = 'pretrained_models/CGAPointNet++_clean.tar'
# MODEL_PATH = 'pretrained_models/PointNet++_clean.tar'
# MODEL_PATH = 'pretrained_models/mlgp_clean.tar'



def rotation_matrix_axis(axis, theta):
    """Generate rotation matrix around a specific axis."""
    c, s = np.cos(theta), np.sin(theta)
    if axis == 'x':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == 'y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif axis == 'z':
        return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])
    else:
        raise ValueError(f"Unknown axis: {axis}")
    
    

def batched_score(model, X, Y, batch_size=128, device='cpu'):
    """Evaluate the score in batches

    Args:
        model: the model to be evaluated
        X    : samples 
        Y    : labels
        batch_size (int, optional): Defaults to 128.
        device (str, optional): Defaults to 'cpu'.

    Returns:
        prediction score
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i+batch_size].to(device)
            Y_batch = Y[i:i+batch_size].to(device)
            preds = model(X_batch).argmax(dim=1)
            correct += (preds == Y_batch).sum().item()
            total += len(Y_batch)
    return correct / total



def evaluate_rotation_robustness(model, Xtest, Ytest, n_trials=10, device='cpu'):
    """Evaluate model under different rotation types."""
    
    test_acc = batched_score(model, Xtest, Ytest, batch_size=128, device=device)
    
    rotation_types = {
        'x-axis': lambda: rotation_matrix_axis('x', np.random.uniform(0, 2*np.pi)),
        'y-axis': lambda: rotation_matrix_axis('y', np.random.uniform(0, 2*np.pi)),
        'z-axis': lambda: rotation_matrix_axis('z', np.random.uniform(0, 2*np.pi)),
        'arbitrary': lambda: random_rotation_matrix(0, 1),
    }
    
    results = {}
    
    with torch.no_grad():
        for rot_name, rot_fn in rotation_types.items():
            accuracies = []
            for _ in range(n_trials):
                Xtest_rot = torch.stack([
                    x @ torch.tensor(rot_fn(), dtype=x.dtype)
                    for x in Xtest
                ])
                accuracies.append(batched_score(model, Xtest_rot, Ytest, batch_size=128, device=device))
            
            results[rot_name] = np.array(accuracies)
    
    # Print results
    print(f'Test accuracy (original):    {test_acc:.5f}\n')
    print(f'{"Rotation":<12} {"Mean":>8} {"Std":>8} {"Drop":>8} {"Min":>8} {"Max":>8}')
    print('-' * 54)
    for rot_name, accs in results.items():
        print(f'{rot_name:<12} {accs.mean():>8.5f} {accs.std():>8.5f} '
              f'{test_acc - accs.mean():>8.5f} {accs.min():>8.5f} {accs.max():>8.5f}')
    
    return test_acc, results
def main():
    print('\nLoading data and model...')
    _, (Xtest, Ytest) = get_model_net_data(
        train_size=1, test_size=908, n_points=1024, 
        class_size=10, force_reload=False, distortion=0.0
    )
    

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_dic = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = model_dic['model'].to(device)
    Xtest, Ytest = Xtest.cpu(), Ytest.cpu()
    
    print(f'\nModel: {model_dic["name"]}')
    evaluate_rotation_robustness(model, Xtest, Ytest, n_trials=20, device=device)
    
if __name__ == '__main__':
    start = time()
    main()
    end = time()
    print(f'Running time: {end- start} s')