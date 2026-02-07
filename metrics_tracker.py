import pandas as pd
import matplotlib.pyplot as plt

class MetricsTracker:
    def __init__(self, model_name):
        self.model_name = model_name
        self.metrics = {
            'epoch': [],
            'loss': [],
            'validation_loss': [],
            'accuracy': [],
            'validation_accuracy': []
        }
    
    def update(self, epoch, loss, val_loss, acc, val_acc):
        self.metrics['epoch'].append(epoch)
        self.metrics['loss'].append(loss)
        self.metrics['validation_loss'].append(val_loss)
        self.metrics['accuracy'].append(acc)
        self.metrics['validation_accuracy'].append(val_acc)
    
    def save(self, filepath=None):
        if filepath is None:
            filepath = f'{self.model_name}_metrics.csv'
        df = pd.DataFrame(self.metrics)
        df.to_csv(filepath, index=False)
    
    @classmethod
    def load(cls, filepath, model_name=None):
        df = pd.read_csv(filepath)
        if model_name is None:
            model_name = filepath.replace('_metrics.csv', '')
        tracker = cls(model_name)
        tracker.metrics = df.to_dict(orient='list')
        return tracker
    
    def plot(self, metric='loss', ax=None, include_validation=True):
        
        # Plot results
        plt.rcParams['lines.linewidth'] = 0.8
        plt.rcParams['font.size'] = 9
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Computer Modern Roman']
        
        
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        
        epochs = self.metrics['epoch']
        ax.plot(epochs, self.metrics[metric], label=f'{self.model_name} {metric}')
        
        if include_validation and f'validation_{metric}' in self.metrics:
            ax.plot(epochs, self.metrics[f'validation_{metric}'], 
                    label=f'{self.model_name} validation {metric}')
        
        return fig, ax