import torch
import numpy as np
from torch.utils.data import Dataset

class RegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ClassificationDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def make_regression_data(n=100, noise=0.1, source='random'):
    if source == 'random':
        X = torch.rand(n, 1)
        w, b = 2.0, -1.0
        y = w * X + b + noise * torch.randn(n, 1)
        return X, y
    elif source == 'diabetes':
        from sklearn.datasets import load_diabetes
        data = load_diabetes()
        X = torch.tensor(data['data'], dtype=torch.float32)
        y = torch.tensor(data['target'], dtype=torch.float32).unsqueeze(1)
        return X, y
    else:
        raise ValueError('Unknown source')

def make_classification_data(n=100, source='random', n_classes=2, n_features=2):
    if source == 'random':
        # Генерация случайных данных для бинарной классификации
        X = torch.rand(n, n_features)
        if n_classes == 2:
            w = torch.randn(n_features, 1)
            b = torch.randn(1)
            logits = X @ w + b
            y = (logits > 0).float()
            return X, y
        else:
            # Для многоклассового случая
            w = torch.randn(n_features, n_classes)
            b = torch.randn(n_classes)
            logits = X @ w + b
            y = torch.softmax(logits, dim=1)
            # Преобразуем в one-hot encoding
            y = torch.zeros(n, n_classes).scatter_(1, torch.argmax(y, dim=1).unsqueeze(1), 1)
            return X, y
            
    elif source == 'breast_cancer':
        # Данные рака груди 
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        X = torch.tensor(data['data'], dtype=torch.float32)
        y = torch.tensor(data['target'], dtype=torch.float32).unsqueeze(1)
        return X, y
        
    elif source == 'multiclass':
        # Генерация многоклассовых данных
        from sklearn.datasets import make_classification
        X, y = make_classification(
            n_samples=n,
            n_features=n_features,
            n_classes=n_classes,
            n_informative=n_features,
            n_redundant=0,
            random_state=42
        )
        X = torch.tensor(X, dtype=torch.float32)
        # Преобразование в one-hot encoding
        y_onehot = torch.zeros(n, n_classes)
        y_onehot.scatter_(1, torch.tensor(y).unsqueeze(1), 1)
        return X, y_onehot
        
    else:
        raise ValueError(f"Unknown source: {source}. Available options: 'random', 'breast_cancer', 'multiclass'")

def mse(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean().item()

def accuracy(y_pred, y_true):
    y_pred_bin = (y_pred > 0.5).float()
    return (y_pred_bin == y_true).float().mean().item()

def log_epoch(epoch, loss, **metrics):
    msg = f"Epoch {epoch}: loss={loss:.4f}"
    for k, v in metrics.items():
        msg += f", {k}={v:.4f}"
    print(msg)