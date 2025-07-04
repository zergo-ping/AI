import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import numpy as np

# Создадим несколько разных датасетов для анализа
datasets = {
    "moons": make_moons(n_samples=10000, noise=0.2, random_state=42),
    "circles": make_circles(n_samples=10000, noise=0.1, factor=0.5, random_state=42),
    "linear": make_classification(n_samples=10000, n_features=2, n_redundant=0, 
                                n_informative=2, n_clusters_per_class=1, random_state=42)
}

# Функция для создания модели с возможностью добавления Dropout и BatchNorm
def create_model(num_layers, use_dropout=False, use_batchnorm=False):
    layers = []
    input_size = 2
    hidden_size = 64
    output_size = 2
    
    for i in range(num_layers):
        output_dim = hidden_size if i < num_layers - 1 else output_size
        layers.append(nn.Linear(input_size, output_dim))
        
        if i < num_layers - 1:  # Не добавляем для выходного слоя
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(output_dim))
            layers.append(nn.ReLU())
            if use_dropout:
                layers.append(nn.Dropout(0.5))
        
        input_size = output_dim
    
    return nn.Sequential(*layers)

# Функция для обучения и оценки модели
def train_and_evaluate(model, train_loader, test_loader, num_epochs=100, lr=0.01):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Обучение
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
    print(f'Training completed in {training_time:.2f} seconds')
    
    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'time': training_time,
        'final_train_acc': train_accs[-1],
        'final_test_acc': test_accs[-1]
    }

# Функция для визуализации
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

# Параметры эксперимента
num_epochs = 150
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Проведем эксперименты для каждого датасета
for dataset_name, (X, y) in datasets.items():
    print(f"\n=== Analyzing dataset: {dataset_name} ===")
    
    # Преобразуем данные в тензоры PyTorch
    X = torch.FloatTensor(X)
    y = torch.LongTensor(y)
    
    # Разделим на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Создадим DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Будем тестировать модели с разным количеством слоев
    layer_configs = [1, 2, 3, 5, 7]
    
    # Словари для хранения результатов
    results_basic = {}
    results_with_dropout = {}
    results_with_batchnorm = {}
    results_with_both = {}
    
    for num_layers in layer_configs:
        print(f"\n--- Testing {num_layers}-layer model ---")
        
        # 1. Базовая модель
        print("Basic model:")
        model = create_model(num_layers)
        res = train_and_evaluate(model, train_loader, test_loader, num_epochs)
        results_basic[num_layers] = res
        plot_results(res, f"{dataset_name} - {num_layers} layers (Basic)")
        
        # 2. Модель с Dropout
        print("\nModel with Dropout:")
        model = create_model(num_layers, use_dropout=True)
        res = train_and_evaluate(model, train_loader, test_loader, num_epochs)
        results_with_dropout[num_layers] = res
        plot_results(res, f"{dataset_name} - {num_layers} layers (Dropout)")
        
        # 3. Модель с BatchNorm
        print("\nModel with BatchNorm:")
        model = create_model(num_layers, use_batchnorm=True)
        res = train_and_evaluate(model, train_loader, test_loader, num_epochs)
        results_with_batchnorm[num_layers] = res
        plot_results(res, f"{dataset_name} - {num_layers} layers (BatchNorm)")
        
        # 4. Модель с Dropout и BatchNorm
        print("\nModel with both Dropout and BatchNorm:")
        model = create_model(num_layers, use_dropout=True, use_batchnorm=True)
        res = train_and_evaluate(model, train_loader, test_loader, num_epochs)
        results_with_both[num_layers] = res
        plot_results(res, f"{dataset_name} - {num_layers} layers (Both)")
    
    # Визуализируем сравнение всех конфигураций
    def plot_comparison(results_dict, title):
        plt.figure(figsize=(15, 6))
        
        for num_layers, res in results_dict.items():
            plt.plot(res['test_accs'], label=f'{num_layers} layers')
        
        plt.title(f'{title} - Test Accuracy Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid()
        plt.show()
    
    print("\n=== Comparison for basic models ===")
    plot_comparison(results_basic, f"{dataset_name} - Basic Models")
    
    print("\n=== Comparison for models with Dropout ===")
    plot_comparison(results_with_dropout, f"{dataset_name} - Models with Dropout")
    
    print("\n=== Comparison for models with BatchNorm ===")
    plot_comparison(results_with_batchnorm, f"{dataset_name} - Models with BatchNorm")
    
    print("\n=== Comparison for models with both techniques ===")
    plot_comparison(results_with_both, f"{dataset_name} - Models with Both")
    
    # Анализ переобучения
    print("\n=== Overfitting Analysis ===")
    for num_layers in layer_configs:
        basic = results_basic[num_layers]
        gap = basic['final_train_acc'] - basic['final_test_acc']
        print(f"{num_layers}-layer basic model: Train-Test gap = {gap:.4f}")
        
        with_both = results_with_both[num_layers]
        gap = with_both['final_train_acc'] - with_both['final_test_acc']
        print(f"{num_layers}-layer with both: Train-Test gap = {gap:.4f}")
    
    # Определим оптимальную глубину для каждого случая
    def find_optimal_depth(results_dict):
        best_test_acc = 0
        best_depth = 1
        for depth, res in results_dict.items():
            if res['final_test_acc'] > best_test_acc:
                best_test_acc = res['final_test_acc']
                best_depth = depth
        return best_depth
    
    print("\n=== Optimal Depth Analysis ===")
    print(f"Basic models optimal depth: {find_optimal_depth(results_basic)}")
    print(f"With Dropout optimal depth: {find_optimal_depth(results_with_dropout)}")
    print(f"With BatchNorm optimal depth: {find_optimal_depth(results_with_batchnorm)}")
    print(f"With both optimal depth: {find_optimal_depth(results_with_both)}")