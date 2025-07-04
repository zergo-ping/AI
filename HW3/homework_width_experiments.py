import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import numpy as np
from prettytable import PrettyTable

# Генерация данных
X, y = make_moons(n_samples=10000, noise=0.2, random_state=42)
X = torch.FloatTensor(X)
y = torch.LongTensor(y)

# Разделение на train и test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Создание DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Устройство для обучения
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Конфигурации ширины слоев
width_configs = {
    "Narrow": [64, 32, 16],
    "Medium": [256, 128, 64],
    "Wide": [1024, 512, 256],
    "Very Wide": [2048, 1024, 512]
}

# Функция для создания модели
def create_model(width_config, use_dropout=False, use_batchnorm=False):
    layers = []
    input_size = 2
    
    for i, hidden_size in enumerate(width_config):
        output_size = hidden_size if i < len(width_config) - 1 else 2
        layers.append(nn.Linear(input_size, output_size))
        
        if i < len(width_config) - 1:  # Не добавляем для выходного слоя
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(output_size))
            layers.append(nn.ReLU())
            if use_dropout:
                layers.append(nn.Dropout(0.5))
        
        input_size = output_size
    
    return nn.Sequential(*layers)

# Функция для подсчета параметров модели
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Функция для обучения и оценки модели
def train_and_evaluate(model, train_loader, test_loader, num_epochs=100, lr=0.001):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = correct_train / total_train
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Тестирование
        model.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
        
        test_loss /= len(test_loader)
        test_acc = correct_test / total_test
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}, '
                  f'Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, '
                  f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    
    training_time = time.time() - start_time
    
    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'time': training_time,
        'final_train_acc': train_accs[-1],
        'final_test_acc': test_accs[-1]
    }

# Функция для визуализации результатов
def plot_results(results, title):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(results['train_losses'], label='Train Loss')
    plt.plot(results['test_losses'], label='Test Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(results['train_accs'], label='Train Accuracy')
    plt.plot(results['test_accs'], label='Test Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Проведем эксперименты для каждой конфигурации
results = {}
param_counts = {}

for name, widths in width_configs.items():
    print(f"\n=== Testing {name} network: {widths} ===")
    
    # Создаем модель
    model = create_model(widths)
    param_count = count_parameters(model)
    param_counts[name] = param_count
    print(f"Number of parameters: {param_count:,}")
    
    # Обучаем модель
    res = train_and_evaluate(model, train_loader, test_loader, num_epochs=100)
    results[name] = res
    
    # Визуализируем результаты
    plot_results(res, f"{name} Network")

# Сравним результаты в таблице
table = PrettyTable()
table.field_names = ["Network Type", "Width Config", "Parameters", "Train Acc", "Test Acc", "Time (s)"]

for name in width_configs.keys():
    table.add_row([
        name,
        width_configs[name],
        f"{param_counts[name]:,}",
        f"{results[name]['final_train_acc']:.4f}",
        f"{results[name]['final_test_acc']:.4f}",
        f"{results[name]['time']:.2f}"
    ])

print("\n=== Comparison of Different Network Widths ===")
print(table)

# Визуализация сравнения всех конфигураций
plt.figure(figsize=(12, 6))
for name, res in results.items():
    plt.plot(res['test_accs'], label=f'{name} (params: {param_counts[name]:,})')

plt.title('Test Accuracy Comparison by Network Width')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()