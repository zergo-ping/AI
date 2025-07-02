import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


class CSVDataset(Dataset):
    def __init__(self, file_path, target_column, 
                 numeric_features=None, categorical_features=None, 
                 binary_features=None, scaling='standard',
                 drop_na=True, na_fill_value=0):
        """
        Кастомный датасет для работы с CSV файлами
        
        Параметры:
        - file_path: путь к CSV файлу
        - target_column: имя целевой колонки
        - numeric_features: список числовых признаков
        - categorical_features: список категориальных признаков
        - binary_features: список бинарных признаков
        - scaling: метод масштабирования ('standard', 'minmax' или None)
        - drop_na: удалять строки с пропущенными значениями
        - na_fill_value: значение для заполнения пропусков
        """
        super().__init__()
        
        # Загрузка данных
        self.data = pd.read_csv(file_path)
        
        # Обработка пропущенных значений
        if drop_na:
            self.data = self.data.dropna()
        else:
            self.data = self.data.fillna(na_fill_value)
        
        # Сохраняем информацию о признаках
        self.target_column = target_column
        self.numeric_features = numeric_features or []
        self.categorical_features = categorical_features or []
        self.binary_features = binary_features or []
        
        # Определяем все используемые признаки
        self.all_features = (
            self.numeric_features + 
            self.categorical_features + 
            self.binary_features
        )
        
        # Проверка наличия целевой колонки
        if target_column not in self.data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Подготовка трансформеров для предобработки
        self._prepare_transformers(scaling)
        
        # Применение предобработки
        self._preprocess_data()
        
    def _prepare_transformers(self, scaling):
        """Создает пайплайны для предобработки данных"""
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler() if scaling == 'standard' else MinMaxScaler())
        ]) if scaling and self.numeric_features else 'passthrough'
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ]) if self.categorical_features else 'passthrough'
        
        # Бинарные признаки не преобразуем
        binary_transformer = 'passthrough'
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features),
                ('bin', binary_transformer, self.binary_features)
            ])
        
    def _preprocess_data(self):
        """Применяет предобработку данных"""
        # Разделяем на признаки и целевую переменную
        X = self.data[self.all_features]
        y = self.data[self.target_column]
        
        # Применяем трансформации к признакам
        self.X_processed = self.preprocessor.fit_transform(X)
        
        # Преобразуем целевую переменную
        if y.dtype == 'object' or len(y.unique()) > 2:
            # Для категориальных целей используем one-hot encoding
            self.y_processed = pd.get_dummies(y).values
            self.problem_type = 'classification'
        else:
            # Для бинарных/числовых целей оставляем как есть
            self.y_processed = y.values.reshape(-1, 1)
            self.problem_type = 'regression' if y.dtype in ['float64', 'int64'] else 'binary'
        
        # Сохраняем имена фичей после преобразования
        self._get_feature_names()
        
    def _get_feature_names(self):
        """Получает имена признаков после преобразования"""
        feature_names = []
        
        # Числовые признаки
        if self.numeric_features:
            feature_names.extend(self.numeric_features)
        
        # Категориальные признаки (развернутые one-hot кодированием)
        if self.categorical_features:
            cat_transformer = self.preprocessor.named_transformers_['cat']
            if hasattr(cat_transformer, 'named_steps'):
                onehot = cat_transformer.named_steps['onehot']
                for i, col in enumerate(self.categorical_features):
                    categories = onehot.categories_[i]
                    feature_names.extend([f"{col}_{cat}" for cat in categories])
        
        # Бинарные признаки
        if self.binary_features:
            feature_names.extend(self.binary_features)
        
        self.feature_names = feature_names
        
    def __len__(self):
        return len(self.X_processed)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.X_processed[idx], dtype=torch.float32)
        y = torch.tensor(self.y_processed[idx], dtype=torch.float32)
        return x, y
    
    def get_feature_names(self):
        """Возвращает имена признаков после преобразования"""
        return self.feature_names
    
    def get_problem_type(self):
        """Возвращает тип задачи (classification/regression/binary)"""
        return self.problem_type