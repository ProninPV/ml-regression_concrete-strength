import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any
from scipy.stats import pearsonr, spearmanr, kurtosis, skew
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ..analysis.eda import calculate_trend_metrics, select_best_transformations
from ..visualization.visualization import plot_feature_trends


def create_feature_analysis(data: pd.DataFrame) -> pd.DataFrame:
    """
    Создает комплексный датафрейм для анализа признаков. Вычисляет коэффициенты Пирсона, Спирмена
    для оригинальных данных, а также значения асимметриb и эксцесса 
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Исходный датафрейм с признаками и целевой переменной
    
    Returns:
    --------
    pandas.DataFrame
        Датафрейм с метриками для каждого признака
    """
    # Проверка на пустые данные
    if data.empty or len(data) == 0 or 'Strength' not in data.columns:
        return pd.DataFrame(columns=['feature', 'pearson_corr', 'spearman_corr', 
                                   'spearman_p_value', 'skewness', 'kurtosis'])
    
    # Извлекаем таргет из исходного датасета
    df = data.drop(columns=["Strength"])
    target_column = data["Strength"]

    # Исключаем ненужные колонки
    features = df.select_dtypes(include='number').columns.tolist()
    
    # Создаем список для хранения результатов
    results = []
    target_data = target_column
    
    for feature in features:
        feature_data = df[feature]
                
        # Базовые статистики
        pearson_corr, pearson_p = pearsonr(feature_data, target_data)
        spearman_corr, spearman_p = spearmanr(feature_data, target_data)
        skewness = skew(feature_data)
        kurtosis_val = kurtosis(feature_data)        
              
        # Собираем результаты
        results.append({
            'feature': feature,
            'pearson_corr': pearson_corr,
            'spearman_corr': spearman_corr,
            'spearman_p_value': round(spearman_p, 4),
            'skewness': skewness,
            'kurtosis': kurtosis_val            
        })
    
    # Создаем датафрейм
    analysis_df = pd.DataFrame(results)
    
    # Проверка перед сортировкой
    if analysis_df.empty:
        return analysis_df
    
    # Сортируем по абсолютной корреляции Спирмена
    analysis_df = analysis_df.sort_values('spearman_corr', ascending=False)
    
    return analysis_df.reset_index(drop=True)


def check_multicollinearity(data: pd.DataFrame,                            
                            threshold: int = 10,
                            plot_heatmap: bool = True) -> pd.DataFrame:
    """
    Проверяет мультиколлинеарность для списка признаков в DataFrame.

    Parameters:
    data: Исходный DataFrame с данными.    
    threshold: Пороговое значение VIF для выделения проблемных признаков. По умолчанию 10.
    plot_heatmap: Строить ли тепловую карту корреляций. По умолчанию True.

    Returns:
    pd.DataFrame: DataFrame с признаками и их VIF значениями, отсортированный по убыванию VIF.
    """
    # Проверка на пустые данные
    if data.empty or len(data) == 0 or 'Strength' not in data.columns:
        return pd.DataFrame(columns=['Feature', 'VIF'])
    
    # Извлекаем таргет из исходного датасета
    df = data.drop(columns=["Strength"])
    target_data = data["Strength"]
    features = df.select_dtypes(include='number').columns.tolist()
        
    # Шаг 1: Матрица корреляций и тепловая карта
    if plot_heatmap:
        # Вычисляем матрицу корреляций только для нужных признаков
        corr_matrix = df[features].corr()
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) # Маска для верхнего треугольника
        sns.heatmap(corr_matrix,
                    mask=mask,
                    annot=True,
                    fmt='.2f',
                    cmap='coolwarm',
                    center=0,
                    square=True)
        plt.title('Матрица корреляций (верхний треугольник)')
        plt.tight_layout()
        plt.show()

    # Шаг 2: Расчет VIF
    # statsmodels требует добавления константы для расчета VIF
    X = add_constant(df[features])
    vif_data = pd.DataFrame()
    vif_data['Feature'] = X.columns
    
    # Рассчитываем VIF для каждого признака
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    # Убираем константу из результатов (ее VIF будет огромным, но она не нужна)
    vif_data = vif_data[vif_data['Feature'] != 'const']
    vif_data = vif_data.sort_values('VIF', ascending=False).reset_index(drop=True)

    return vif_data


def highlight_high_vif(vif_data: pd.DataFrame, 
                       high_threshold: int = 10, 
                       medium_threshold: int = 5):
    """
    Визуально выделяет признаки с высоким VIF в таблице с градацией цветов.
    
    Parameters:
        vif_data (pd.DataFrame): DataFrame с колонками ['Feature', 'VIF']
        high_threshold (int): Порог для красного выделения (VIF >= 10). По умолчанию 10
        medium_threshold (int): Порог для желтого выделения (5 <= VIF < 10). По умолчанию 5
    
    Returns:
        pd.io.formats.style.Styler: Стилизованный DataFrame для отображения
    """
    
    def _highlight_vif_gradient(s):
        """Внутренняя функция для применения градиентного стиля"""
        styles = []
        for vif_value in s:
            if vif_value >= high_threshold:
                styles.append('background-color: #ffcccc')
            elif medium_threshold <= vif_value < high_threshold:
                styles.append('background-color: #fff2cc')
            else:
                styles.append('')
        return styles
    
    styled_vif = (vif_data.style
                  .apply(_highlight_vif_gradient, subset=['VIF'])
                  .format({'VIF': '{:.2f}'}))
    
    return styled_vif


def plot_feature_trends_orchestrator(df: pd.DataFrame,
                                     config: Dict[str, Any],
                                     target: str = 'Strength',
                                     figsize: Tuple[int, int] = (16, 12),
                                     alpha: float = 0.2) -> pd.DataFrame:
    """
    Основная функция-оркестратор для построения трендов признаков.
    """
    features = df.select_dtypes(include=[np.number]).columns
    features = [f for f in features if f != target]
    
    metrics_df = calculate_trend_metrics(df, features, target, config)
    results_df = select_best_transformations(metrics_df, alpha)   

    # Используем функцию из visualization
    plot_feature_trends(df, results_df, features, target, config, figsize) 
    
    return results_df