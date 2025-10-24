import os
import yaml
import logging
import numpy as np
import scipy.stats as stats
import sys
import warnings
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import levene
from typing import List, Any, Optional, Tuple, Dict
from datetime import datetime
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr, kurtosis, skew
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.base import BaseEstimator, TransformerMixin


class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Обработчик выбросов для использования в sklearn Pipeline.
    Поддерживает различные стратегии обработки выбросов.
    """
    
    def __init__(self, 
                 strategies: List[str] = ['abnormal'],
                 config: Optional[Dict] = None,
                 target_col: str = None):
        """
        Инициализация обработчика выбросов.
        
        Args:
            strategies: Список стратегий обработки 
                      ['abnormal', 'combine', 'gost_binar', 'gost_remove', 'iqr_remove']
            config: Словарь с конфигурацией ГОСТ диапазонов и аномальных значений
            target_col: Название целевой переменной (например, 'Strength')
        """
        self.strategies = strategies
        self.config = config or self._get_default_config()
        self.target_col = target_col
        self.iqr_bounds_ = {}
        self.feature_names_in_ = None
        self.fitted_ = False
        self.removed_indices_ = set()
        self.outlier_summary_ = {}
        
        # Инициализация порогов из конфигурации
        self._init_thresholds()
    
    def _get_default_config(self):
        """Конфигурация по умолчанию если не предоставлена"""
        return {
            'standard_value': {
                'age': {'max': 365, 'min': 1},
                'cement': {'max': 600, 'min': 200},
                'coarse_aggregate': {'max': 1300, 'min': 1000},
                'fine_aggregate': {'max': 800, 'min': 600},
                'fly_ash': {'max': 200, 'min': 0},
                'sp_c_pct': {'max': 0.025, 'min': 0.005},
                'strength': {'max': 100, 'min': 5},
                'w_c': {'max': 0.7, 'min': 0.3},
                'water': {'max': 220, 'min': 120}
            },
            'abnormal_value': {
                'sp_c_pct': 0.07,
                'fine_aggregate': 970
            }
        }
    
    def _init_thresholds(self):
        """Инициализация порогов из конфигурации"""
        # ГОСТ диапазоны
        self.gost_ranges = self.config.get('standard_value', {})
        
        # Аномальные значения
        abnormal_config = self.config.get('abnormal_value', {})
        self.abnormal_thresholds = {
            'sp_c_pct': abnormal_config.get('sp_c_pct', 0.07),
            'fine_aggregate': abnormal_config.get('fine_aggregate', 970)
        }
        
        # Бинарные пороги для combine стратегии
        self.combine_thresholds = {
            'sp_critical': self.abnormal_thresholds['sp_c_pct'],
            'sp_warning': self.gost_ranges.get('sp_c_pct', {}).get('max', 0.025),
            'fa_critical': self.abnormal_thresholds['fine_aggregate'],
            'fa_warning': self.gost_ranges.get('fine_aggregate', {}).get('max', 800)
        }
    
    def fit(self, X, y=None):
        """
        Обучение обработчика на данных.
        
        Args:
            X: DataFrame с признаками
            y: Целевая переменная (опционально)
            
        Returns:
            self: Обученный трансформер
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        self.feature_names_in_ = X.columns.tolist()
        
        # Сохранение индексов для синхронизации с y
        self.original_indices_ = X.index
        
        # Расчет IQR границ если нужна стратегия iqr_remove
        if 'iqr_remove' in self.strategies:
            self._calculate_iqr_bounds(X)
        
        # Инициализация сводки
        self._init_outlier_summary(X)
        
        self.fitted_ = True
        return self
    
    def _calculate_iqr_bounds(self, X):
        """Расчет IQR границ для числовых признаков"""
        numeric_features = X.select_dtypes(include=[np.number]).columns
        
        for feature in numeric_features:
            Q1 = X[feature].quantile(0.25)
            Q3 = X[feature].quantile(0.75)
            IQR = Q3 - Q1
            
            self.iqr_bounds_[feature] = {
                'lower': Q1 - 1.5 * IQR,
                'upper': Q3 + 1.5 * IQR,
                'q1': Q1,
                'q3': Q3
            }
    
    def _init_outlier_summary(self, X):
        """Инициализация сводки по выбросам"""
        self.outlier_summary_ = {
            'original_shape': X.shape,
            'strategies_applied': [],
            'removed_rows': 0,
            'binary_features_created': [],
            'outliers_by_feature': {},
            'removed_indices': set()
        }
    
    def transform(self, X, y=None):
        """
        Применение обработки выбросов к данным.
        
        Args:
            X: DataFrame с признаками
            y: Целевая переменная (опционально)
            
        Returns:
            X_processed: Обработанный DataFrame
            y_processed: Обработанная целевая переменная (если y предоставлена)
        """
        if not self.fitted_:
            raise ValueError("Сначала необходимо вызвать fit()")
            
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        X_processed = X.copy()
        y_processed = y.copy() if y is not None else None
        
        # Сброс сводки
        self.removed_indices_ = set()
        self._init_outlier_summary(X)
        
        # Применение стратегий в указанном порядке
        for strategy in self.strategies:
            if strategy == 'abnormal':
                X_processed, y_processed = self._apply_abnormal_strategy(X_processed, y_processed)
            elif strategy == 'combine':
                X_processed, y_processed = self._apply_combine_strategy(X_processed, y_processed)
            elif strategy == 'gost_binar':
                X_processed = self._apply_gost_binar_strategy(X_processed)
            elif strategy == 'gost_remove':
                X_processed, y_processed = self._apply_gost_remove_strategy(X_processed, y_processed)
            elif strategy == 'iqr_remove':
                X_processed, y_processed = self._apply_iqr_remove_strategy(X_processed, y_processed)
            else:
                print(f"Неизвестная стратегия: {strategy}")
        
        # Обновление сводки
        self.outlier_summary_['final_shape'] = X_processed.shape
        self.outlier_summary_['removed_rows'] = len(self.removed_indices_)
        self.outlier_summary_['removed_indices'] = self.removed_indices_
        
        if y is not None:
            return X_processed, y_processed
        return X_processed
    
    def _apply_abnormal_strategy(self, X, y=None):
        """Стратегия abnormal: удаление критических выбросов"""
        rows_to_remove = set()
        
        # Superplasticizer >= порог удалить строки
        sp_col = 'Sp/C_pct' if 'Sp/C_pct' in X.columns else 'sp_c_pct'
        if sp_col in X.columns:
            sp_threshold = self.abnormal_thresholds['sp_c_pct']
            sp_outliers = X[X[sp_col] >= sp_threshold].index
            rows_to_remove.update(sp_outliers)
            self._update_outlier_summary('Sp/C_pct', len(sp_outliers), sp_threshold)
        
        # Fine Aggregate >= порог удалить строки
        fa_col = 'Fine Aggregate' if 'Fine Aggregate' in X.columns else 'fine_aggregate'
        if fa_col in X.columns:
            fa_threshold = self.abnormal_thresholds['fine_aggregate']
            fa_outliers = X[X[fa_col] > fa_threshold].index
            rows_to_remove.update(fa_outliers)
            self._update_outlier_summary('Fine Aggregate', len(fa_outliers), fa_threshold)
        
        return self._remove_rows(X, y, rows_to_remove, 'abnormal')
    
    def _apply_combine_strategy(self, X, y=None):
        """Стратегия combine: комбинированная обработка"""
        rows_to_remove = set()
        
        # Обработка Superplasticizer
        sp_col = 'Sp/C_pct' if 'Sp/C_pct' in X.columns else 'sp_c_pct'
        if sp_col in X.columns:
            # Удаление строк с SP >= критический порог
            sp_critical = self.combine_thresholds['sp_critical']
            sp_remove = X[X[sp_col] >= sp_critical].index
            rows_to_remove.update(sp_remove)
            
            # Бинарный признак для SP >= предупреждающий порог
            sp_warning = self.combine_thresholds['sp_warning']
            X['High_SP'] = (X[sp_col] >= sp_warning).astype(int)
            high_sp_count = X['High_SP'].sum()
            
            self.outlier_summary_['binary_features_created'].append('High_SP')
            self._update_outlier_summary('High_SP', high_sp_count, sp_warning)
        
        # Обработка Fine Aggregate
        fa_col = 'Fine Aggregate' if 'Fine Aggregate' in X.columns else 'fine_aggregate'
        if fa_col in X.columns:
            # Удаление строк с FA >= критический порог
            fa_critical = self.combine_thresholds['fa_critical']
            fa_remove = X[X[fa_col] >= fa_critical].index
            rows_to_remove.update(fa_remove)
            
            # Бинарный признак для FA >= предупреждающий порог
            fa_warning = self.combine_thresholds['fa_warning']
            X['High_FA'] = (X[fa_col] >= fa_warning).astype(int)
            high_fa_count = X['High_FA'].sum()
            
            self.outlier_summary_['binary_features_created'].append('High_FA')
            self._update_outlier_summary('High_FA', high_fa_count, fa_warning)
        
        return self._remove_rows(X, y, rows_to_remove, 'combine')
    
    def _apply_gost_binar_strategy(self, X):
        """Стратегия gost_binar: бинарные признаки по ГОСТ"""
        if not self.gost_ranges:
            print("GOST binar: Конфигурация ГОСТ не найдена")
            return X
        
        for feature, ranges in self.gost_ranges.items():
            # Поиск соответствующего столбца в данных
            data_col = self._find_matching_column(X, feature)
            if data_col:
                min_val = ranges.get('min')
                max_val = ranges.get('max')
                
                if min_val is not None and max_val is not None:
                    # Бинарный признак для значений вне ГОСТ диапазона
                    outlier_mask = (X[data_col] < min_val) | (X[data_col] > max_val)
                    binary_feature_name = f'Outlier_{feature}'
                    X[binary_feature_name] = outlier_mask.astype(int)
                    
                    outlier_count = outlier_mask.sum()
                    if outlier_count > 0:
                        self.outlier_summary_['binary_features_created'].append(binary_feature_name)
                        self._update_outlier_summary(feature, outlier_count, f"{min_val}-{max_val}")
        
        return X
    
    def _apply_gost_remove_strategy(self, X, y=None):
        """Стратегия gost_remove: удаление по ГОСТ"""
        if not self.gost_ranges:
            print("GOST remove: Конфигурация ГОСТ не найдена")
            return X, y
            
        rows_to_remove = set()
        
        for feature, ranges in self.gost_ranges.items():
            # Поиск соответствующего столбца в данных
            data_col = self._find_matching_column(X, feature)
            if data_col:
                min_val = ranges.get('min')
                max_val = ranges.get('max')
                
                if min_val is not None and max_val is not None:
                    outliers = X[(X[data_col] < min_val) | (X[data_col] > max_val)].index
                    rows_to_remove.update(outliers)
                    
                    if outliers.any():
                        self._update_outlier_summary(feature, len(outliers), f"{min_val}-{max_val}")
        
        return self._remove_rows(X, y, rows_to_remove, 'gost_remove')
    
    def _apply_iqr_remove_strategy(self, X, y=None):
        """Стратегия iqr_remove: удаление по IQR"""
        if not self.iqr_bounds_:
            print("IQR remove: IQR границы не рассчитаны")
            return X, y
            
        rows_to_remove = set()
        
        for feature, bounds in self.iqr_bounds_.items():
            if feature in X.columns:
                outliers = X[
                    (X[feature] < bounds['lower']) | 
                    (X[feature] > bounds['upper'])
                ].index
                rows_to_remove.update(outliers)
                
                if outliers.any():
                    self._update_outlier_summary(
                        f"IQR_{feature}", 
                        len(outliers), 
                        f"{bounds['lower']:.2f}-{bounds['upper']:.2f}"
                    )
        
        return self._remove_rows(X, y, rows_to_remove, 'iqr_remove')
    
    def _find_matching_column(self, X, feature_name):
        """Поиск соответствующего столбца в данных"""
        # Прямое соответствие
        if feature_name in X.columns:
            return feature_name
        
        # Поиск похожих названий
        possible_matches = {
            'cement': ['Cement', 'cement'],
            'water': ['Water', 'water'],
            'fine_aggregate': ['Fine Aggregate', 'FineAggregate', 'fine_aggregate'],
            'coarse_aggregate': ['Coarse Aggregate', 'CoarseAggregate', 'coarse_aggregate'],
            'fly_ash': ['Fly Ash', 'FlyAsh', 'fly_ash'],
            'sp_c_pct': ['Sp/C_pct', 'Superplasticizer_pct', 'sp_c_pct'],
            'w_c': ['W/C', 'Water_Cement_Ratio', 'w_c'],
            'age': ['Age', 'age'],
            'strength': ['Strength', 'strength']
        }
        
        if feature_name in possible_matches:
            for match in possible_matches[feature_name]:
                if match in X.columns:
                    return match
        
        return None
    
    def _remove_rows(self, X, y, rows_to_remove, strategy_name):
        """Удаление строк из X и y"""
        if rows_to_remove:
            self.removed_indices_.update(rows_to_remove)
            self.outlier_summary_['strategies_applied'].append(
                f"{strategy_name}: {len(rows_to_remove)} rows"
            )
            
            X_processed = X.drop(index=rows_to_remove)
            y_processed = y.drop(index=rows_to_remove) if y is not None else None
            
            return X_processed, y_processed
        
        return X, y
    
    def _update_outlier_summary(self, feature, count, threshold):
        """Обновление сводки по выбросам"""
        if feature not in self.outlier_summary_['outliers_by_feature']:
            self.outlier_summary_['outliers_by_feature'][feature] = []
        
        self.outlier_summary_['outliers_by_feature'][feature].append({
            'count': count,
            'threshold': threshold
        })
    
    def fit_transform(self, X, y=None):
        """Обучение и преобразование в одном методе"""
        return self.fit(X, y).transform(X, y)
    
    def get_feature_names_out(self, input_features=None):
        """Получение имен признаков после преобразования"""
        if not self.fitted_:
            raise ValueError("Сначала необходимо вызвать fit()")
        return self.feature_names_in_
    
    def get_outlier_summary(self):
        """Получение сводки по обработке выбросов"""
        return self.outlier_summary_
    
    def visualize_outliers(self, X, features=None, figsize=(15, 10)):
        """
        Визуализация выбросов до и после обработки.
        
        Args:
            X: Исходные данные
            features: Список признаков для визуализации
            figsize: Размер фигуры
        """
        if not self.fitted_:
            raise ValueError("Сначала необходимо вызвать fit()")
        
        if features is None:
            # Выбор числовых признаков для визуализации
            numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
            features = numeric_features[:6]  # Ограничим 6 признаками
        
        n_features = len(features)
        n_cols = 2
        n_rows = (n_features + 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for i, feature in enumerate(features):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Данные до обработки
            data_before = X[feature]
            
            # Данные после обработки
            X_processed = self.transform(X)
            data_after = X_processed[feature] if feature in X_processed.columns else pd.Series([])
            
            # Построение boxplot
            data_to_plot = [data_before.dropna()]
            labels = ['Before']
            
            if len(data_after) > 0:
                data_to_plot.append(data_after.dropna())
                labels.append('After')
            
            ax.boxplot(data_to_plot, labels=labels)
            ax.set_title(f'Outliers: {feature}')
            ax.set_ylabel('Values')
            
            # Добавление информации о выбросах
            outliers_before = self._count_outliers_iqr(data_before)
            outliers_after = self._count_outliers_iqr(data_after) if len(data_after) > 0 else 0
            
            ax.text(0.5, 0.95, f'Outliers: {outliers_before}', 
                   transform=ax.transAxes, ha='center', va='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            
            if len(data_after) > 0:
                ax.text(1.5, 0.95, f'Outliers: {outliers_after}', 
                       transform=ax.transAxes, ha='center', va='top',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        
        # Скрываем пустые subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        # Вывод сводки
        self._print_processing_summary()
    
    def _count_outliers_iqr(self, data):
        """Подсчет выбросов по IQR"""
        if len(data) == 0:
            return 0
        
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        return ((data < lower_bound) | (data > upper_bound)).sum()
    
    def _print_processing_summary(self):
        """Вывод сводки по обработке"""
        summary = self.get_outlier_summary()
        
        print("=" * 50)
        print("OUTLIER PROCESSING SUMMARY")
        print("=" * 50)
        print(f"Original data shape: {summary['original_shape']}")
        print(f"Final data shape: {summary.get('final_shape', 'N/A')}")
        print(f"Rows removed: {summary['removed_rows']}")
        print(f"Applied strategies: {', '.join(summary['strategies_applied'])}")
        
        if summary['binary_features_created']:
            print(f"Binary features created: {', '.join(summary['binary_features_created'])}")
        
        print("\nOutliers by feature:")
        for feature, outliers in summary['outliers_by_feature'].items():
            for outlier_info in outliers:
                print(f"  - {feature}: {outlier_info['count']} (threshold: {outlier_info['threshold']})")
        
        print("=" * 50)


class ZeroBinaryEncoder(BaseEstimator, TransformerMixin):
    """
    Класс для добавления бинарных признаков на основе значимости нулевых значений
    """
    
    def __init__(self, alpha: float = 0.05, levene_alpha: float = 0.05):
        """
        Инициализация класса
        
        Parameters:
        -----------
        alpha : float
            Уровень значимости для t-теста
        levene_alpha : float
            Уровень значимости для теста Левене на равенство дисперсий
        """
        self.alpha = alpha
        self.levene_alpha = levene_alpha
        self.significant_features_ = []
        self.test_results_ = {}
        self.binary_features_created_ = []
        
    def fit(self, X_preprocessed: pd.DataFrame, y: pd.Series) -> 'BinaryFeatureSignificance':
        """
        Анализ значимости нулевых значений для признаков
        
        Parameters:
        -----------
        X_preprocessed : pd.DataFrame
            Обработанные признаки
        y : pd.Series
            Целевая переменная
            
        Returns:
        --------
        self
        """
        # Признаки для анализа (можно расширить)
        features_to_analyze = ['Blast Furnace Slag', 'Fly Ash', 'Sp/C_pct']
        
        for feature in features_to_analyze:
            if feature not in X_preprocessed.columns:
                warnings.warn(f"Признак {feature} не найден в данных. Пропускаем.")
                continue
                
            # Создаем бинарный признак: 1 если значение > 0, 0 если == 0
            binary_feature = (X_preprocessed[feature] > 0).astype(int)
            
            # Разделяем целевую переменную на две группы
            group_non_zero = y[binary_feature == 1]
            group_zero = y[binary_feature == 0]
            
            # Проверяем достаточно ли данных в группах
            if len(group_non_zero) < 2 or len(group_zero) < 2:
                warnings.warn(f"Недостаточно данных для анализа признака {feature}")
                self.test_results_[feature] = {
                    'p_value': 1.0,
                    'levene_p_value': np.nan,
                    'test_type': "Недостаточно данных",
                    'significant': False,
                    'group_non_zero_size': len(group_non_zero),
                    'group_zero_size': len(group_zero),
                    'group_non_zero_mean': np.nan,
                    'group_zero_mean': np.nan
                }
                continue
            
            # Тест Левене на равенство дисперсий
            levene_p = np.nan
            equal_var = False
            
            try:
                # Проверяем, что в обеих группах есть хотя бы 2 различных значения
                if (group_non_zero.nunique() >= 2 and group_zero.nunique() >= 2 and
                    len(group_non_zero) >= 2 and len(group_zero) >= 2):
                    levene_stat, levene_p = levene(group_non_zero, group_zero)  # Теперь определен
                    equal_var = levene_p > self.levene_alpha
                else:
                    levene_p = np.nan
                    equal_var = False
                    test_type = "Welch's t-test (недостаточно данных для Левене)"
            except (ValueError, ZeroDivisionError, RuntimeWarning) as e:
                levene_p = np.nan
                equal_var = False
                test_type = f"Welch's t-test (ошибка Левене: {str(e)[:30]})"
            
            # t-тест
            try:
                if equal_var and not np.isnan(levene_p):
                    t_stat, t_p = stats.ttest_ind(group_non_zero, group_zero, 
                                                 equal_var=True)
                    test_type = "Student's t-test (дисперсии равны)"
                else:
                    t_stat, t_p = stats.ttest_ind(group_non_zero, group_zero, 
                                                 equal_var=False)
                    test_type = "Welch's t-test (дисперсии не равны)"
            except (ValueError, ZeroDivisionError, RuntimeWarning) as e:
                t_p = 1.0
                test_type = f"Тест не выполнен (ошибка: {str(e)[:30]})"
            
            # Сохраняем результаты
            self.test_results_[feature] = {
                'p_value': t_p,
                'levene_p_value': levene_p,
                'test_type': test_type,
                'significant': t_p < self.alpha,
                'group_non_zero_size': len(group_non_zero),
                'group_zero_size': len(group_zero),
                'group_non_zero_mean': group_non_zero.mean(),
                'group_zero_mean': group_zero.mean()
            }
            
            if t_p < self.alpha:
                self.significant_features_.append(feature)
                
        return self
    
    def transform(self, X_preprocessed: pd.DataFrame) -> pd.DataFrame:
        """
        Добавление бинарных признаков
        
        Parameters:
        -----------
        X_preprocessed : pd.DataFrame
            Обработанные признаки
            
        Returns:
        --------
        pd.DataFrame
            Данные с добавленными бинарными признаками
        """
        X_transformed = X_preprocessed.copy()
        self.binary_features_created_ = []
        
        for feature in self.significant_features_:
            if feature in X_preprocessed.columns:
                binary_feature_name = f"{feature}_binary"
                X_transformed[binary_feature_name] = (X_preprocessed[feature] > 0).astype(int)
                self.binary_features_created_.append(binary_feature_name)
                
        return X_transformed
    
    def fit_transform(self, X_preprocessed: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Объединенный метод fit и transform
        
        Parameters:
        -----------
        X_preprocessed : pd.DataFrame
            Обработанные признаки
        y : pd.Series
            Целевая переменная
            
        Returns:
        --------
        pd.DataFrame
            Данные с добавленными бинарными признаками
        """
        return self.fit(X_preprocessed, y).transform(X_preprocessed)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Получение сводки по анализу
        
        Returns:
        --------
        Dict
            Словарь с результатами анализа
        """
        return {
            'significant_features': self.significant_features_,
            'test_results': self.test_results_,
            'binary_features_created': self.binary_features_created_,
            'alpha': self.alpha,
            'levene_alpha': self.levene_alpha
        }
    
            
        # 4. Сводная таблица
        summary_data = []
        for feature in features:
            result = self.test_results_[feature]
            summary_data.append([
                feature,
                f"{result['p_value']:.2e}",
                '✓' if result['significant'] else '✗',
                result['test_type'].split(' ')[0],
                f"{result['levene_p_value']:.3f}" if not np.isnan(result['levene_p_value']) else 'N/A'
            ])
        
        ax4.axis('off')
        table = ax4.table(cellText=summary_data,
                         colLabels=['Признак', 'p-value', 'Значим', 'Тест', 'Левене p'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        ax4.set_title('Сводка по тестам')
        
        plt.tight_layout()
        return fig
    
    def print_detailed_report(self):
        """Вывод детального отчета в стиле примера"""
        print("Анализ значимости нулевых значений для признаков:")
        print("=" * 60)
        
        for feature, result in self.test_results_.items():
            status = "✓" if result['significant'] else "✗"
            action = "Добавляем бинарный признак" if result['significant'] else "Признак не добавляем"
            
            print(f"{status} {feature}: p-value = {result['p_value']:.3f} - {action}")
            print(f"  {result['test_type']}, p-value Левене = {result['levene_p_value']:.3f}")
            print("-" * 40)


class FeatureHandleEngineering(BaseEstimator, TransformerMixin):
    """
    Класс для создания инженерных признаков на основе состава бетона
    """
    
    def __init__(self, wc_column='W/C'):
        self.wc_column = wc_column
        self.low_wc_threshold_ = None
        self.high_wc_threshold_ = None
        self.feature_names_ = None
        
    def fit(self, X, y=None):
        """
        Вычисляет пороги для адаптивных признаков W/C ratio
        """
        # Проверяем, что необходимые признаки присутствуют
        required_features = ['Cement', 'Blast Furnace Slag', 'Fly Ash', 
                           'Coarse Aggregate', 'Fine Aggregate', 'Water']
        
        missing_features = [feat for feat in required_features if feat not in X.columns]
        if missing_features:
            raise ValueError(f"Отсутствуют необходимые признаки: {missing_features}")
        
        # Вычисляем адаптивные пороги для W/C ratio
        if self.wc_column in X.columns:
            self.low_wc_threshold_ = X[self.wc_column].quantile(0.25)
            self.high_wc_threshold_ = X[self.wc_column].quantile(0.75)
        else:
            # Если W/C нет, вычисляем его
            wc_ratio = X['Water'] / X['Cement']
            self.low_wc_threshold_ = wc_ratio.quantile(0.25)
            self.high_wc_threshold_ = wc_ratio.quantile(0.75)
        
        # Сохраняем имена признаков для совместимости с sklearn
        self.feature_names_ = list(X.columns)
        
        return self
    
    def transform(self, X):
        """
        Создает новые инженерные признаки
        """
        X_transformed = X.copy()
        
        # 1. Бинарные признаки наличия компонентов
        X_transformed['is_zero_slag'] = (X_transformed['Blast Furnace Slag'] == 0).astype(int)
        X_transformed['is_zero_superplastic'] = (X_transformed['Superplasticizer'] == 0).astype(int)
        
        # 2. Композитные признаки
        X_transformed['Total_powder'] = (X_transformed['Cement'] + 
                                       X_transformed['Blast Furnace Slag'] + 
                                       X_transformed['Fly Ash'])
        
        # Защита от деления на ноль
        X_transformed['Aggregate_ratio'] = np.where(
            X_transformed['Fine Aggregate'] > 0,
            X_transformed['Coarse Aggregate'] / X_transformed['Fine Aggregate'],
            0
        )
        
        X_transformed['Binder_water_ratio'] = np.where(
            X_transformed['Water'] > 0,
            X_transformed['Total_powder'] / X_transformed['Water'],
            0
        )
        
        # 3. Признаки W/C ratio
        if self.wc_column in X_transformed.columns:
            wc_ratio = X_transformed[self.wc_column]
        else:
            # Если W/C нет в данных, вычисляем его
            wc_ratio = X_transformed['Water'] / X_transformed['Cement']
            X_transformed['W/C'] = wc_ratio
        
        # Адаптивные пороги (вычисленные в fit)
        X_transformed['Low_WC_ratio'] = (wc_ratio < self.low_wc_threshold_).astype(int)
        X_transformed['High_WC_ratio'] = (wc_ratio > self.high_wc_threshold_).astype(int)
        
        # Технологические пороги
        X_transformed['Low_WC_tech'] = (wc_ratio < 0.4).astype(int)
        X_transformed['High_WC_tech'] = (wc_ratio > 0.6).astype(int)
        
        return X_transformed
    
    def get_feature_summary(self):
        """
        Возвращает сводку по созданным признакам
        """
        return {
            'total_features_created': 9,
            'features_created': [
                'is_zero_slag', 'is_zero_superplastic', 'Total_powder',
                'Aggregate_ratio', 'Binder_water_ratio', 'Low_WC_ratio',
                'High_WC_ratio', 'Low_WC_tech', 'High_WC_tech'
            ],
            'wc_thresholds': {
                'low_adaptive': self.low_wc_threshold_,
                'high_adaptive': self.high_wc_threshold_,
                'low_tech': 0.4,
                'high_tech': 0.6
            }
        }
    
    def get_feature_descriptions(self):
        """
        Возвращает описание созданных признаков
        """
        return {
            'is_zero_slag': 'Наличие шлака в смеси (0/1)',
            'is_zero_superplastic': 'Наличие суперпластификатора (0/1)',
            'Total_powder': 'Общее количество вяжущего (цемент + шлак + зола)',
            'Aggregate_ratio': 'Соотношение крупного и мелкого заполнителя',
            'Binder_water_ratio': 'Соотношение вяжущего и воды',
            'Low_WC_ratio': 'Низкое В/Ц отношение (адаптивный порог)',
            'High_WC_ratio': 'Высокое В/Ц отношение (адаптивный порог)',
            'Low_WC_tech': 'Низкое В/Ц отношение (< 0.4)',
            'High_WC_tech': 'Высокое В/Ц отношение (> 0.6)'
        }
    

class FeatureUninformRemove(BaseEstimator, TransformerMixin):
    """
    Transformer для удаления неинформативных признаков.
    Удаляет признаки, где доля одного значения превышает заданный порог.
    
    Parameters
    ----------
    threshold : float, default=0.95
        Порог для удаления признака (доля наиболее частого значения)
    verbose : bool, default=True
        Вывод детальной статистики
    """
    
    def __init__(self, threshold=0.95, verbose=True):
        self.threshold = threshold
        self.verbose = verbose
        self.columns_to_drop_ = []
        self.removal_stats_ = {}
        
    def fit(self, X, y=None):
        """
        Определяет признаки для удаления на основе дисбаланса значений.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Входные данные
        y : array-like, optional
            Целевая переменная (не используется)
            
        Returns
        -------
        self : object
            Возвращает self
        """
        if isinstance(X, pd.DataFrame):
            X_analysis = X
        else:
            X_analysis = pd.DataFrame(X)
            
        self.columns_to_drop_ = []
        self.removal_stats_ = {}
        
        for col in X_analysis.columns:
            value_counts = X_analysis[col].value_counts(normalize=True)
            max_ratio = value_counts.iloc[0] if len(value_counts) > 0 else 0
            total_values = len(X_analysis[col])
            
            if max_ratio > self.threshold:
                self.columns_to_drop_.append(col)
                self.removal_stats_[col] = {
                    'max_value_ratio': max_ratio,
                    'total_values': total_values,
                    'most_frequent_value': value_counts.index[0],
                    'most_frequent_count': int(value_counts.iloc[0] * total_values)
                }
        
        if self.verbose:
            self._print_removal_stats()
            
        return self
    
    def transform(self, X):
        """
        Удаляет неинформативные признаки.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Входные данные
            
        Returns
        -------
        X_transformed : array-like
            Данные с удаленными признаками
        """
        if isinstance(X, pd.DataFrame):
            X_transformed = X.drop(columns=self.columns_to_drop_, errors='ignore')
        else:
            X_transformed = pd.DataFrame(X).drop(columns=self.columns_to_drop_, errors='ignore')
            
        if self.verbose and len(self.columns_to_drop_) > 0:
            print(f"\nУдалено признаков: {len(self.columns_to_drop_)}")
            print(f"Оставлено признаков: {X_transformed.shape[1]}")
            
        return X_transformed
    
    def _print_removal_stats(self):
        """Выводит детальную статистику об удаляемых признаках."""
        if not self.removal_stats_:
            print("Неинформативные признаки не обнаружены.")
            return
            
        print("=" * 60)
        print("АНАЛИЗ НЕИНФОРМАТИВНЫХ ПРИЗНАКОВ")
        print("=" * 60)
        print(f"Порог для удаления: {self.threshold:.1%}")
        print(f"Всего признаков для удаления: {len(self.columns_to_drop_)}")
        print("-" * 60)
        
        for i, (col, stats) in enumerate(self.removal_stats_.items(), 1):
            print(f"{i}. Признак: {col}")
            print(f"   - Максимальная доля значения: {stats['max_value_ratio']:.3f} ({stats['max_value_ratio']:.1%})")
            print(f"   - Самое частое значение: {stats['most_frequent_value']}")
            print(f"   - Количество повторений: {stats['most_frequent_count']}/{stats['total_values']}")
            print(f"   - Причина удаления: доля > {self.threshold:.1%}")
            print("-" * 40)
    
    def get_feature_names_out(self, input_features=None):
        """
        Возвращает имена признаков после трансформации.
        
        Parameters
        ----------
        input_features : array-like of str, optional
            Имена входных признаков
            
        Returns
        -------
        feature_names_out : ndarray of str objects
            Имена выходных признаков
        """
        if input_features is None:
            if hasattr(self, 'feature_names_in_'):
                input_features = self.feature_names_in_
            else:
                raise ValueError("Необходимо передать input_features или обучить трансформер")
        
        remaining_features = [f for f in input_features if f not in self.columns_to_drop_]
        return np.array(remaining_features)
    
    def get_removal_summary(self):
        """
        Возвращает сводку по удаленным признакам.
        
        Returns
        -------
        summary : dict
            Словарь с информацией об удаленных признаках
        """
        return {
            'threshold': self.threshold,
            'columns_removed': self.columns_to_drop_,
            'removal_stats': self.removal_stats_,
            'total_removed': len(self.columns_to_drop_)
        }


class CollinearityReducer(BaseEstimator, TransformerMixin):
    """
    Улучшенный класс для обработки мультиколлинеарности с более мягким подходом
    """
    
    def __init__(self, vif_threshold=10, correlation_threshold=0.95,
                 priority_strategy='domain_priority', 
                 domain_priority_list=None,
                 protected_features=None,  # Защищенные признаки которые НЕЛЬЗЯ удалять
                 max_removal_percentage=0.3,  # Максимум 30% признаков можно удалить
                 verbose=True):
        
        self.vif_threshold = vif_threshold
        self.correlation_threshold = correlation_threshold
        self.priority_strategy = priority_strategy
        self.domain_priority_list = domain_priority_list or []
        self.protected_features = protected_features or ['Water', 'Cement', 'Age']
        self.max_removal_percentage = max_removal_percentage
        self.verbose = verbose
        
        # Атрибуты для хранения результатов
        self.removed_features_ = []
        self.final_features_ = []
        self.vif_report_ = {}
        self.correlation_report_ = {}
        self.feature_names_ = []
    
    def fit(self, X, y=None):
        """Обучение трансформера"""
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X_work = X.copy()
        else:
            self.feature_names_ = [f'feature_{i}' for i in range(X.shape[1])]
            X_work = pd.DataFrame(X, columns=self.feature_names_)
        
        self.removed_features_ = []
        
        # Основной анализ мультиколлинеарности
        self._fit_improved(X_work)
        
        if self.verbose:
            self.print_report()
            
        return self
    
    def _fit_improved(self, X):
        """Улучшенный подход к анализу мультиколлинеарности"""
        initial_feature_count = len(X.columns)
        max_features_to_remove = int(initial_feature_count * self.max_removal_percentage)
        
        # Шаг 1: Итеративное удаление по VIF с ограничением
        features_to_remove_vif = self._iterative_vif_removal_improved(X, max_features_to_remove)
        self.removed_features_.extend(features_to_remove_vif)
        
        # Проверяем, не превысили ли лимит удаления
        if len(self.removed_features_) >= max_features_to_remove:
            if self.verbose:
                print(f"⚠️  Достигнут лимит удаления ({self.max_removal_percentage*100}%)")
            self.final_features_ = [f for f in X.columns if f not in self.removed_features_]
            return
        
        # Шаг 2: Дополнительная проверка на экстремальные корреляции
        remaining_features = [f for f in X.columns if f not in self.removed_features_]
        X_remaining = X[remaining_features]
        
        if len(X_remaining.columns) > 1:
            extreme_corr_removals = self._find_extreme_correlations_improved(X_remaining, 
                                                                           max_features_to_remove - len(self.removed_features_))
            self.removed_features_.extend(extreme_corr_removals)
        
        self.final_features_ = [f for f in X.columns if f not in self.removed_features_]
    
    def _iterative_vif_removal_improved(self, X, max_removals):
        """Улучшенное итеративное удаление с защитой важных признаков"""
        features_to_remove = []
        current_features = X.columns.tolist()
        X_current = X.copy()
        
        iteration = 0
        vif_history = []
        
        while (len(current_features) > 1 and 
               len(features_to_remove) < max_removals and
               len(features_to_remove) < len(current_features) - 1):  # Минимум 1 признак должен остаться
            
            iteration += 1
            
            # Вычисляем VIF для всех признаков
            vif_scores = self._calculate_vif_scores(X_current)
            vif_history.append(vif_scores.copy())
            
            # Находим кандидатов для удаления (VIF > threshold), исключая защищенные
            high_vif_features = {f: v for f, v in vif_scores.items() 
                               if v > self.vif_threshold and f not in self.protected_features}
            
            if not high_vif_features:
                break
                
            # Выбор стратегии удаления (только из незащищенных признаков)
            feature_to_remove = self._select_feature_to_remove_improved(high_vif_features, current_features, X_current)
            
            if feature_to_remove and feature_to_remove not in self.protected_features:
                features_to_remove.append(feature_to_remove)
                current_features.remove(feature_to_remove)
                X_current = X_current[current_features]
            else:
                break  # Не нашли подходящий признак для удаления
        
        # Сохраняем отчет по VIF
        self.vif_report_ = {
            'initial_vif': vif_history[0] if vif_history else {},
            'final_vif': vif_scores,
            'iterations': iteration,
            'vif_history': vif_history
        }
                
        return features_to_remove
    
    def _select_feature_to_remove_improved(self, high_vif_features, current_features, X_current):
        """Улучшенный выбор признака для удаления"""
        # Фильтруем только незащищенные признаки
        available_features = {f: v for f, v in high_vif_features.items() 
                            if f not in self.protected_features and f in current_features}
        
        if not available_features:
            return None
            
        if self.priority_strategy == 'vif_based':
            return max(available_features.items(), key=lambda x: x[1])[0]
            
        elif self.priority_strategy == 'domain_priority':
            return self._remove_by_domain_priority_improved(available_features, current_features)
            
        elif self.priority_strategy == 'statistical':
            return self._remove_by_statistical_priority(available_features, X_current)
        else:
            return max(available_features.items(), key=lambda x: x[1])[0]
    
    def _remove_by_domain_priority_improved(self, high_vif_features, current_features):
        """Улучшенное удаление по доменному приоритету"""
        # Создаем полный список приоритетов (защищенные + доменные + остальные)
        full_priority_list = self.protected_features + self.domain_priority_list
        
        prioritized_candidates = []
        
        # Сначала добавляем признаки из доменного списка (кроме защищенных)
        for feature in full_priority_list:
            if (feature in high_vif_features and 
                feature in current_features and 
                feature not in self.protected_features):
                prioritized_candidates.append(feature)
        
        # Затем добавляем остальные признаки
        for feature in high_vif_features:
            if (feature not in prioritized_candidates and 
                feature in current_features and
                feature not in self.protected_features):
                prioritized_candidates.append(feature)
        
        # Удаляем ПОСЛЕДНИЙ в списке (наименее важный)
        return prioritized_candidates[-1] if prioritized_candidates else None
    
    def _find_extreme_correlations_improved(self, X, max_removals):
        """Улучшенный поиск экстремальных корреляций"""
        features_to_remove = []
        corr_matrix = X.corr().abs()
        
        upper_triangle = corr_matrix.where(
            np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        )
        
        # Находим экстремальные пары
        extreme_pairs = []
        for i in range(len(upper_triangle.columns)):
            for j in range(i):
                corr_value = upper_triangle.iloc[i, j]
                if not np.isnan(corr_value) and corr_value > self.correlation_threshold:
                    feature_i = upper_triangle.columns[i]
                    feature_j = upper_triangle.columns[j]
                    extreme_pairs.append((feature_i, feature_j, corr_value))
        
        # Сохраняем отчет по корреляциям
        self.correlation_report_ = {
            'extreme_pairs': extreme_pairs,
            'correlation_matrix': corr_matrix
        }
        
        # Удаляем по одному признаку из каждой экстремальной пары (только незащищенные)
        removed_in_step = []
        for feature_i, feature_j, corr_value in extreme_pairs:
            if (len(features_to_remove) < max_removals and
                feature_i not in features_to_remove and 
                feature_j not in features_to_remove):
                
                # Выбираем какой признак удалить (предпочитаем незащищенные)
                candidates = []
                if feature_i not in self.protected_features:
                    candidates.append(feature_i)
                if feature_j not in self.protected_features:
                    candidates.append(feature_j)
                
                if candidates:
                    # Используем доменные приоритеты для выбора
                    feature_to_remove = self._select_from_candidates_by_priority(candidates)
                    features_to_remove.append(feature_to_remove)
                    removed_in_step.append(feature_to_remove)
                
        return features_to_remove
    
    def _select_from_candidates_by_priority(self, candidates):
        """Выбор кандидата для удаления на основе приоритетов"""
        if not candidates:
            return None
            
        # Создаем полный список приоритетов
        full_priority_list = self.protected_features + self.domain_priority_list
        
        # Сортируем кандидатов по приоритету (чем выше индекс - тем менее важен)
        sorted_candidates = sorted(candidates, 
                                 key=lambda x: full_priority_list.index(x) if x in full_priority_list 
                                 else len(full_priority_list))
        
        # Возвращаем наименее важный (последний в отсортированном списке)
        return sorted_candidates[-1]
    
    def _calculate_vif_scores(self, X):
        """Вычисление VIF для всех признаков"""
        vif_scores = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i, feature in enumerate(X.columns):
                try:
                    vif = variance_inflation_factor(X.values, i)
                    vif_scores[feature] = vif if not np.isinf(vif) else 1000
                except:
                    vif_scores[feature] = 1000
        return vif_scores
    
    def _remove_by_statistical_priority(self, high_vif_features, X_current):
        """Удаление по статистическим критериям"""
        std_scores = {}
        for feature in high_vif_features:
            if feature in X_current.columns and feature not in self.protected_features:
                std_scores[feature] = X_current[feature].std()
        
        if not std_scores:
            return None
            
        return min(std_scores.items(), key=lambda x: x[1])[0]
    
    def transform(self, X, y=None):
        """Применение трансформации"""
        if not hasattr(self, 'final_features_'):
            raise ValueError("Необходимо сначала вызвать метод fit()")
        
        if isinstance(X, pd.DataFrame):
            return X[self.final_features_]
        else:
            feature_indices = [self.feature_names_.index(col) for col in self.final_features_]
            return X[:, feature_indices]
    
    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y).transform(X, y)
    
    def get_removal_report(self):
        """Получение детального отчета"""
        initial_features = len(self.feature_names_)
        final_features = len(self.final_features_)
        
        report = {
            'parameters': {
                'vif_threshold': self.vif_threshold,
                'correlation_threshold': self.correlation_threshold,
                'priority_strategy': self.priority_strategy,
                'domain_priority_list': self.domain_priority_list,
                'protected_features': self.protected_features,
                'max_removal_percentage': self.max_removal_percentage
            },
            'summary': {
                'initial_features': initial_features,
                'final_features': final_features,
                'removed_features_count': len(self.removed_features_),
                'removal_percentage': (len(self.removed_features_) / initial_features) * 100,
                'protected_features_remaining': len([f for f in self.final_features_ if f in self.protected_features])
            },
            'removed_features': self.removed_features_,
            'final_features': self.final_features_,
            'vif_analysis': self.vif_report_,
            'correlation_analysis': self.correlation_report_
        }
        
        return report
    
    def print_report(self):
        """Вывод подробного отчета в консоль"""
        report = self.get_removal_report()
        
        print("=" * 70)
        print("УЛУЧШЕННЫЙ ОТЧЕТ ПО МУЛЬТИКОЛЛИНЕАРНОСТИ")
        print("=" * 70)
        
        print(f"\nПараметры анализа:")
        print(f"  - VIF порог: {report['parameters']['vif_threshold']}")
        print(f"  - Порог корреляции: {report['parameters']['correlation_threshold']}")
        print(f"  - Стратегия: {report['parameters']['priority_strategy']}")
        print(f"  - Защищенные признаки: {report['parameters']['protected_features']}")
        print(f"  - Максимальное удаление: {report['parameters']['max_removal_percentage']*100}%")
        
        print(f"\nРезультаты:")
        print(f"  - Исходное количество признаков: {report['summary']['initial_features']}")
        print(f"  - Финальное количество признаков: {report['summary']['final_features']}")
        print(f"  - Удалено признаков: {report['summary']['removed_features_count']}")
        print(f"  - Процент удаления: {report['summary']['removal_percentage']:.1f}%")
        print(f"  - Защищенных признаков сохранено: {report['summary']['protected_features_remaining']}")
        
        if self.removed_features_:
            print(f"\nУдаленные признаки:")
            for feature in self.removed_features_:
                protection_status = "🛡️ ЗАЩИЩЕН" if feature in self.protected_features else "❌ УДАЛЕН"
                print(f"  - {feature}: {protection_status}")
        
        # Информация о защищенных признаках
        protected_remaining = [f for f in self.final_features_ if f in self.protected_features]
        if protected_remaining:
            print(f"\n🛡️ Сохраненные защищенные признаки:")
            for feature in protected_remaining:
                print(f"  - {feature}")
        
        print(f"\nОставшиеся признаки ({len(self.final_features_)}):")
        print(f"  {self.final_features_}")
        print("=" * 70)


class FeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, config, target_col, alpha=0.2):
        self.config = config
        self.target_col = target_col
        self.alpha = alpha
        self.best_transformations_ = {}
        self.transformation_report_ = {}
        
        self.transform_functions = {
            'Linear': lambda x: x,
            'Log': lambda x: np.log(np.maximum(x, 1e-8)),
            'Sqrt': lambda x: np.sqrt(np.maximum(x, 0)),
            '1/x': lambda x: 1 / np.where(x == 0, 1e-8, np.abs(x)),
            'square_func': lambda x: x ** 2
        }
        
        self.trend_mapping = {
            'Linear': 'Linear',
            'Log': 'Log', 
            'Sqrt': 'Sqrt',
            '1/x': '1/x',
            'square_func': 'x²'
        }
        
        # Паттерны для идентификации бинарных признаков
        self.binary_patterns = [
            'Outlier_', 'High_', 'Low_', 'is_zero', 'is_', 
            'has_', 'flag_', 'binary_', '_bin', '_flag'
        ]
    
    def _is_binary_feature(self, feature_name, feature_data):
        """Проверяет, является ли признак бинарным"""
        # Проверка по имени
        name_lower = feature_name.lower()
        if any(pattern.lower() in name_lower for pattern in self.binary_patterns):
            return True
        
        # Проверка по данным - если только 2 уникальных значения
        unique_vals = np.unique(feature_data)
        if len(unique_vals) <= 2:
            return True
            
        return False
    
    def _safe_transform(self, func, x, feature_name, transform_name):
        """Безопасное применение преобразования с диагностикой"""
        try:
            result = func(x)
            
            if np.any(np.isinf(result)) or np.any(np.isnan(result)):
                return None
            if np.std(result) == 0:
                return None
                
            return result
        except Exception as e:
            return None
    
    def _calculate_r2(self, X_transformed, y):
        """Вычисление R2 score БЕЗ LinearRegression - через прямую формулу"""
        try:
            if X_transformed is None:
                return -np.inf
                
            # Проверка на валидность данных
            if (np.any(np.isinf(X_transformed)) or 
                np.any(np.isnan(X_transformed)) or
                np.std(X_transformed) == 0 or
                len(np.unique(X_transformed)) <= 1):
                return -np.inf
            
            # Прямое вычисление R2 через корреляцию
            correlation = np.corrcoef(X_transformed, y)[0, 1]
            
            if np.isnan(correlation) or np.isinf(correlation):
                return -np.inf
                
            r2 = correlation ** 2
            
            return r2
            
        except Exception as e:
            return -np.inf
    
    def fit(self, X, y=None):
        if y is None:
            if self.target_col in X.columns:
                y = X[self.target_col].values
                X = X.drop(columns=[self.target_col])
            else:
                raise ValueError("Target column not found in X and y not provided")
        
        feature_names = X.columns
        transformation_names = self.config['trend_settings']['names']
        
        print("🔍 FeatureTransformer: вычисляем R2 через корреляцию...")
        
        for feature in feature_names:
            feature_data = X[feature].values
            
            # ⭐⭐⭐ ПРОВЕРКА НА БИНАРНЫЙ ПРИЗНАК ⭐⭐⭐
            if self._is_binary_feature(feature, feature_data):
                print(f"\n📊 Анализ признака: {feature} [БИНАРНЫЙ]")
                print(f"   min={feature_data.min():.2f}, max={feature_data.max():.2f}")
                print(f"   ⭐ Для бинарных признаков всегда используем Linear")
                
                # Для бинарных признаков всегда используем Linear
                self.best_transformations_[feature] = 'Linear'
                self.transformation_report_[feature] = {
                    'best_transformation': 'Linear',
                    'r2_scores': {'Linear': -np.inf},  # Не вычисляем R2 для бинарных
                    'best_r2': -np.inf,
                    'is_binary': True
                }
                continue
            
            # Обычная обработка для небинарных признаков
            r2_scores = {}
            
            print(f"\n📊 Анализ признака: {feature}")
            print(f"   min={feature_data.min():.2f}, max={feature_data.max():.2f}")
            
            # Проверяем исходные данные
            if np.any(np.isnan(feature_data)) or np.any(np.isinf(feature_data)):
                print(f"   ❌ Исходные данные содержат NaN/inf")
                continue
            
            # Вычисляем R2 для каждого преобразования
            for transform_name in transformation_names:
                if transform_name in self.transform_functions:
                    # Применяем преобразование
                    transformed_data = self._safe_transform(
                        self.transform_functions[transform_name], 
                        feature_data,
                        feature,
                        transform_name
                    )
                    
                    # Вычисляем R2
                    r2 = self._calculate_r2(transformed_data, y)
                    r2_scores[transform_name] = r2
                    
                    if r2 > -np.inf:
                        print(f"   ✅ {transform_name}: R2 = {r2:.4f}")
                    else:
                        print(f"   ❌ {transform_name}: невалидно")
            
            # Если все R2 = -inf, используем Linear по умолчанию
            if all(r2 == -np.inf for r2 in r2_scores.values()):
                print(f"   ⚠️ Все преобразования невалидны, используем Linear")
                best_transform = 'Linear'
                best_r2 = -np.inf
            else:
                # Используем ваш алгоритм выбора
                best_transform, best_r2 = self._select_best_transformation(r2_scores)
                print(f"   🎯 Лучшее: {best_transform} (R2={best_r2:.4f})")
            
            # Сохраняем лучшее преобразование
            self.best_transformations_[feature] = best_transform
            self.transformation_report_[feature] = {
                'best_transformation': self.trend_mapping.get(best_transform, best_transform),
                'r2_scores': {self.trend_mapping.get(k, k): v for k, v in r2_scores.items()},
                'best_r2': best_r2,
                'is_binary': False
            }
        
        return self
    
    def _select_best_transformation(self, r2_scores):
        """Ваш алгоритм выбора лучшего преобразования"""
        # Фильтруем только валидные R2
        valid_scores = {k: v for k, v in r2_scores.items() if v > -np.inf}
        
        if not valid_scores:
            return 'Linear', -np.inf
        
        # Преобразуем ключи в ваши trend_names
        trend_names_mapping = {
            'Linear': 'Linear',
            'Log': 'Log',
            'Sqrt': 'Sqrt', 
            '1/x': '1/x',
            'square_func': 'x²'
        }
        
        mapped_scores = {}
        for transform_name, r2 in valid_scores.items():
            if transform_name in trend_names_mapping:
                mapped_name = trend_names_mapping[transform_name]
                mapped_scores[mapped_name] = r2
        
        # Ваш оригинальный алгоритм
        best_trend_name = max(mapped_scores.items(), key=lambda x: x[1])[0]
        best_r2 = mapped_scores[best_trend_name]
        linear_r2 = mapped_scores.get('Linear', -np.inf)
        
        # Обратное преобразование к нашим именам функций
        reverse_mapping = {v: k for k, v in trend_names_mapping.items()}
        
        if best_trend_name != 'Linear' and linear_r2 >= best_r2 * (1 - self.alpha):
            final_best_trend = 'Linear'
            final_best_r2 = linear_r2
        else:
            final_best_trend = best_trend_name
            final_best_r2 = best_r2
        
        return reverse_mapping.get(final_best_trend, 'Linear'), final_best_r2
    
    def transform(self, X):
        X_transformed = X.copy()
        
        target_present = False
        if self.target_col in X_transformed.columns:
            target_data = X_transformed[self.target_col]
            X_transformed = X_transformed.drop(columns=[self.target_col])
            target_present = True
        
        for feature, transform_name in self.best_transformations_.items():
            if feature in X_transformed.columns:
                transform_func = self.transform_functions[transform_name]
                transformed_values = transform_func(X_transformed[feature].values)
                X_transformed[feature] = transformed_values
        
        if target_present:
            X_transformed[self.target_col] = target_data
        
        return X_transformed
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
    
    def get_transformation_report(self):
        report = []
        for feature, info in self.transformation_report_.items():
            report.append({
                'Признак': feature,
                'Лучшее преобразование': info['best_transformation'],
                'R2 score': f"{info['best_r2']:.6f}",
                'Все R2 scores': {k: f"{v:.6f}" for k, v in info['r2_scores'].items()},
                'Тип': 'Бинарный' if info.get('is_binary', False) else 'Небинарный'
            })
        return pd.DataFrame(report)
    
    def print_report(self):
        df_report = self.get_transformation_report()
        print("\n" + "="*80)
        print("ФИНАЛЬНЫЙ ОТЧЕТ")
        print("="*80)
        
        # Разделяем бинарные и небинарные признаки
        binary_features = df_report[df_report['Тип'] == 'Бинарный']
        non_binary_features = df_report[df_report['Тип'] == 'Небинарный']
        
        if len(non_binary_features) > 0:
            print("\n📈 НЕБИНАРНЫЕ ПРИЗНАКИ (с преобразованиями):")
            for _, row in non_binary_features.iterrows():
                print(f"   {row['Признак']} -> {row['Лучшее преобразование']} (R2={row['R2 score']})")
        
        if len(binary_features) > 0:
            print(f"\n🔘 БИНАРНЫЕ ПРИЗНАКИ ({len(binary_features)} шт., всегда Linear):")
            binary_list = [row['Признак'] for _, row in binary_features.iterrows()]
            print(f"   {', '.join(binary_list)}")
        
        # Статистика
        total_transformed = len(non_binary_features[non_binary_features['Лучшее преобразование'] != 'Linear'])
        print(f"\n📊 СТАТИСТИКА:")
        print(f"   Всего признаков: {len(df_report)}")
        print(f"   Бинарных: {len(binary_features)}")
        print(f"   Небинарных: {len(non_binary_features)}")
        print(f"   Преобразовано: {total_transformed}")
        
        return df_report