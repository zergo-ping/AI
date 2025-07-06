
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import seaborn as sns

from models.cnn_models import *
from models.fc_models import *


batch_size_MNIST = 64
epochs_MNIST = 10
learning_rate_MNIST = 0.001

# Загрузка данных MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset_MNIST = torchvision.datasets.MNIST(root='./data_MNIST', train=True, transform=transform, download=True)
test_dataset_MNIST = torchvision.datasets.MNIST(root='./data_MNIST', train=False, transform=transform)

train_loader_MNIST = DataLoader(train_dataset_MNIST, batch_size=batch_size_MNIST, shuffle=True)
test_loader_MNIST = DataLoader(test_dataset_MNIST, batch_size=batch_size_MNIST, shuffle=False)


# Функция для обучения
def train_model(model, train_loader, test_loader, epochs, learning_rate):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    times = []
    
    model.to(device)
    
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Тестирование
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        test_loss /= len(test_loader)
        test_acc = 100. * correct / total
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        epoch_time = time.time() - start_time
        times.append(epoch_time)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, '
              f'Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%, Time: {epoch_time:.2f}s')
    
    # Измерение времени инференса
    inference_time = 0.0
    model.eval()
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            start = time.time()
            _ = model(inputs)
            inference_time += time.time() - start
    inference_time /= len(test_loader.dataset) / batch_size_MNIST
    
    return train_losses, test_losses, train_accs, test_accs, times, inference_time

# Создание моделей
models = {
    'FCN': FCN(),
    'SimpleCNN': SimpleCNN(),
    'ResCNN': ResCNN()
}

results = {}

# Обучение моделей
for name, model in models.items():
    print(f'\nTraining {name}')
    train_losses, test_losses, train_accs, test_accs, times, inference_time = train_model(
        model, train_loader_MNIST, test_loader_MNIST, epochs_MNIST, learning_rate_MNIST)
    results[name] = {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'times': times,
        'inference_time': inference_time,
        'params': sum(p.numel() for p in model.parameters())
    }


# Настройки
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 128
epochs = 10
learning_rate = 0.001
# Загрузка данных CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



# Функция для обучения с дополнительной аналитикой
def train_cifar_model(model, train_loader, test_loader, epochs, learning_rate):

    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    times = []
    gradients = {name: [] for name, _ in model.named_parameters() if 'weight' in name}
    
    model.to(device)
    
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Запись градиентов
            for name, param in model.named_parameters():
                if 'weight' in name and param.grad is not None:
                    gradients[name].append(param.grad.norm().item())
            
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Тестирование
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        test_loss /= len(test_loader)
        test_acc = 100. * correct / total
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        # Confusion matrix для последней эпохи
        if epoch == epochs - 1:
            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {model.__class__.__name__}')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.show()
        
        epoch_time = time.time() - start_time
        times.append(epoch_time)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, '
              f'Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%, Time: {epoch_time:.2f}s')
    
    # Визуализация градиентов
    plt.figure(figsize=(12, 6))
    for name, grad in gradients.items():
        plt.plot(grad, label=name)
    plt.title(f'{model.__class__.__name__}s')
    plt.xlabel('Iteration')
    plt.ylabel('Gradient Norm')
    plt.legend()
    plt.show()
    
    return train_losses, test_losses, train_accs, test_accs, times

# Создание моделей
cifar_models = {
    'DeepFCN': DeepFCN(),
    'ResNetCIFAR': ResNetCIFAR(),
    'RegularizedResNet': RegularizedResNet()
}

cifar_results = {}

# Обучение моделей
for name, model in cifar_models.items():
    print(f'\nTraining {name}')
    train_losses, test_losses, train_accs, test_accs, times = train_cifar_model(
        model, train_loader, test_loader, epochs, learning_rate)
    cifar_results[name] = {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'times': times,
        'params': sum(p.numel() for p in model.parameters())
    }