import torch
from torch.utils.data import DataLoader
from utils import ClassificationDataset, accuracy, make_classification_data, make_regression_data, mse, log_epoch, RegressionDataset
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class LinearRegressionManual:
    def __init__(self, in_features, l1_lambda=0.01, l2_lambda=0.01):
        self.w = torch.randn(in_features, 1, dtype=torch.float32, requires_grad=False)
        self.b = torch.zeros(1, dtype=torch.float32, requires_grad=False)
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

    def __call__(self, X):
        return X @ self.w + self.b

    def parameters(self):
        return [self.w, self.b]

    def zero_grad(self):
        self.dw = torch.zeros_like(self.w)
        self.db = torch.zeros_like(self.b)

    def backward(self, X, y, y_pred):
        n = X.shape[0]
        error = y_pred - y
        
        # Основные градиенты
        self.dw = (X.T @ error) / n
        self.db = error.mean(0)
        
        # Добавляем регуляризацию
        if self.l1_lambda > 0:
            self.dw += self.l1_lambda * torch.sign(self.w)
        if self.l2_lambda > 0:
            self.dw += self.l2_lambda * 2 * self.w

    def step(self, lr):
        self.w -= lr * self.dw
        self.b -= lr * self.db

    def save(self, path):
        torch.save({'w': self.w, 'b': self.b}, path)

    def load(self, path):
        state = torch.load(path)
        self.w = state['w']
        self.b = state['b']

def train_with_early_stopping(model, dataloader, lr=0.1, max_epochs=100, patience=5, tol=1e-4):
    best_loss = float('inf')
    epochs_no_improve = 0
    best_params = None
    
    for epoch in range(1, max_epochs + 1):
        total_loss = 0
        
        for i, (batch_X, batch_y) in enumerate(dataloader):
            y_pred = model(batch_X)
            loss = mse(y_pred, batch_y)
            total_loss += loss
            
            model.zero_grad()
            model.backward(batch_X, batch_y, y_pred)
            model.step(lr)
        
        avg_loss = total_loss / (i + 1)
        
        # Early stopping логика
        if avg_loss < best_loss - tol:
            best_loss = avg_loss
            epochs_no_improve = 0
            best_params = (model.w.clone(), model.b.clone())
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch}')
                if best_params is not None:
                    model.w, model.b = best_params
                break
        
        if epoch % 10 == 0:
            log_epoch(epoch, avg_loss)
    
    return model

if __name__ == '__main__':
    # Генерируем данные
    X, y = make_regression_data(n=200)
    
    # Создаём датасет и даталоадер
    dataset = RegressionDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f'Размер датасета: {len(dataset)}')
    print(f'Количество батчей: {len(dataloader)}')
    print(f'Пример данных: {dataset[0]}')
    
    # Обучаем модель с регуляризацией и early stopping
    model = LinearRegressionManual(in_features=1, l1_lambda=0.01, l2_lambda=0.01)
    model = train_with_early_stopping(model, dataloader, lr=0.1, max_epochs=1000, patience=10)
    
    model.save('HW2\models\linreg_manual.pth')

# Модифицируйте существующую логистическую регрессию:
# - Добавьте поддержку многоклассовой классификации
# - Реализуйте метрики: precision, recall, F1-score, ROC-AUC
# - Добавьте визуализацию confusion matrix


def softmax(x):
    exp_x = torch.exp(x - torch.max(x, dim=1, keepdim=True).values)
    return exp_x / exp_x.sum(dim=1, keepdim=True)

class LogisticRegressionManual:
    def __init__(self, in_features, n_classes=2):
        # Инициализация весов для многоклассовой классификации
        self.w = torch.randn(in_features, n_classes, dtype=torch.float32, requires_grad=False)
        self.b = torch.zeros(n_classes, dtype=torch.float32, requires_grad=False)
        self.n_classes = n_classes

    def __call__(self, X):
        # Применяем softmax для многоклассовой классификации
        logits = X @ self.w + self.b
        return softmax(logits)

    def parameters(self):
        return [self.w, self.b]

    def zero_grad(self):
        self.dw = torch.zeros_like(self.w)
        self.db = torch.zeros_like(self.b)

    def backward(self, X, y, y_pred):
        n = X.shape[0]
        # Градиент для кросс-энтропийной потери с softmax
        error = y_pred - y
        self.dw = (X.T @ error) / n
        self.db = error.mean(0)

    def step(self, lr):
        self.w -= lr * self.dw
        self.b -= lr * self.db

    def save(self, path):
        torch.save({'w': self.w, 'b': self.b, 'n_classes': self.n_classes}, path)

    def load(self, path):
        state = torch.load(path)
        self.w = state['w']
        self.b = state['b']
        self.n_classes = state['n_classes']

def calculate_metrics(y_true, y_pred, y_probs, n_classes):
    """Вычисление метрик качества классификации"""
    metrics = {}
    
    # Для бинарной классификации
    if n_classes == 2:
        metrics['precision'] = precision_score(y_true, y_pred)
        metrics['recall'] = recall_score(y_true, y_pred)
        metrics['f1'] = f1_score(y_true, y_pred)
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_probs[:, 1])
        except:
            metrics['roc_auc'] = 0.5
    # Для многоклассовой классификации
    else:
        metrics['precision'] = precision_score(y_true, y_pred, average='macro')
        metrics['recall'] = recall_score(y_true, y_pred, average='macro')
        metrics['f1'] = f1_score(y_true, y_pred, average='macro')
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_probs, multi_class='ovo')
        except:
            metrics['roc_auc'] = 0.5
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, n_classes):
    """Визуализация матрицы ошибок"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(n_classes), 
                yticklabels=range(n_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == '__main__':
    # Генерируем данные для многоклассовой классификации
    n_classes = 3
    X, y = make_classification_data(n=200, n_classes=n_classes)
    
    # Создаём датасет и даталоадер
    dataset = ClassificationDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f'Размер датасета: {len(dataset)}')
    print(f'Количество батчей: {len(dataloader)}')
    print(f'Пример данных: {dataset[0]}')
    
    # Обучаем модель
    model = LogisticRegressionManual(in_features=2, n_classes=n_classes)
    lr = 0.1
    epochs = 100
    
    # Для хранения метрик
    all_metrics = []
    
    for epoch in range(1, epochs + 1):
        total_loss = 0
        total_acc = 0
        all_y = []
        all_y_pred = []
        all_y_probs = []
        
        for i, (batch_X, batch_y) in enumerate(dataloader):
            # Прямой проход
            y_probs = model(batch_X)
            
            # Преобразование one-hot в метки классов
            y_true = torch.argmax(batch_y, dim=1)
            y_pred = torch.argmax(y_probs, dim=1)
            
            # Вычисление потерь (кросс-энтропия)
            loss = -torch.sum(batch_y * torch.log(y_probs + 1e-8)) / batch_X.shape[0]
            acc = accuracy(y_probs, batch_y)
            
            total_loss += loss.item()
            total_acc += acc
            
            # Сохраняем для метрик
            all_y.extend(y_true.numpy())
            all_y_pred.extend(y_pred.numpy())
            all_y_probs.extend(y_probs.detach().numpy())
            
            # Обратный проход и обновление весов
            model.zero_grad()
            model.backward(batch_X, batch_y, y_probs)
            model.step(lr)
        
        # Вычисление метрик
        metrics = calculate_metrics(all_y, all_y_pred, np.array(all_y_probs), n_classes)
        all_metrics.append(metrics)
        
        avg_loss = total_loss / (i + 1)
        avg_acc = total_acc / (i + 1)
        
        if epoch % 10 == 0:
            log_epoch(epoch, avg_loss, acc=avg_acc)
            print(f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, "
                  f"F1: {metrics['f1']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # Визуализация матрицы ошибок
    plot_confusion_matrix(all_y, all_y_pred, n_classes)
    
    # Сохранение модели
    model.save('HW2\models\logreg_manual.pth')