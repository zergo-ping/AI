# Визуализация результатов
from matplotlib import pyplot as plt
import numpy as np

from utils.train import *



def vizualize_first_task():

    plt.figure(figsize=(12, 8))

    # Графики потерь
    plt.subplot(2, 2, 1)
    for name in models:
        plt.plot(results[name]['train_losses'], label=f'{name} Train')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    for name in models:
        plt.plot(results[name]['test_losses'], label=f'{name} Test')
    plt.title('Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Графики точности
    plt.subplot(2, 2, 3)
    for name in models:
        plt.plot(results[name]['train_accs'], label=f'{name} Train')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.subplot(2, 2, 4)
    for name in models:
        plt.plot(results[name]['test_accs'], label=f'{name} Test')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Анализ параметров и времени
    for name in models:
        print(f'{name}:')
        print(f'  Параметры: {results[name]["params"]}')
        print(f'  Время обучения за эпоху: {np.mean(results[name]["times"]):.2f}s')
        print(f'  Время вывода за пакет: {results[name]["inference_time"]*1000:.2f}ms')
        print(f'  Итоговая точность обучения: {results[name]["train_accs"][-1]:.2f}%')
        print(f'  Итоговая точность тестирования: {results[name]["test_accs"][-1]:.2f}%')


def vizualize_second_task():
    # Визуализация результатов
    plt.figure(figsize=(12, 8))

    # Графики потерь
    plt.subplot(2, 2, 1)
    for name in cifar_models:
        plt.plot(cifar_results[name]['train_losses'], label=f'{name} Train')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    for name in cifar_models:
        plt.plot(cifar_results[name]['test_losses'], label=f'{name} Test')
    plt.title('Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Графики точности
    plt.subplot(2, 2, 3)
    for name in cifar_models:
        plt.plot(cifar_results[name]['train_accs'], label=f'{name} Train')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.subplot(2, 2, 4)
    for name in cifar_models:
        plt.plot(cifar_results[name]['test_accs'], label=f'{name} Test')
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Анализ параметров и времени
    for name in cifar_models:
        print(f'{name}:')
        print(f'  Параметры: {cifar_results[name]["params"]}')
        print(f'  Время обучения за эпоху: {np.mean(cifar_results[name]["times"]):.2f}s')
        print(f'  Итоговая точность обучения: {cifar_results[name]["train_accs"][-1]:.2f}%')
        print(f'  Итоговая точность тестирования: {cifar_results[name]["test_accs"][-1]:.2f}%')
        print(f'  Промежуток между тренировкой и тестированием: {cifar_results[name]["train_accs"][-1] - cifar_results[name]["test_accs"][-1]:.2f}%')