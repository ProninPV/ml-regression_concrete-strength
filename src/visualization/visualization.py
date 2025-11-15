import os
import yaml
import logging
import numpy as np
import importlib
import subprocess
import scipy.stats as stats
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Any, Optional, Tuple, Dict
from datetime import datetime
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr, kurtosis, skew
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


def plot_outliers(df: pd.DataFrame,
                  summary_df: pd.DataFrame,
                  max_plots: int = 20) -> None:
    """
    Строит boxplot-графики для признаков с наибольшим числом выбросов.

    max_plots — ограничение на количество отображаемых графиков
    """
    for _, row in summary_df.head(max_plots).iterrows():
        col = row['feature']
        plt.figure(figsize=(6, 1.5))
        sns.boxplot(x=df[col], color='skyblue')
        plt.title(f"Boxplot: {col} (IQR выбросов: {row['n_outliers_IQR']})")
        plt.tight_layout()
        plt.show()


def visualize_feature_analysis(analysis_df: pd.DataFrame) -> None:
    """
    Визуализирует сравнение корреляций Пирсона и Спирмена для каждого признака с целевой переменной и
    значения асимметрии и эксцесса для признаков.
    
    Parameters:
    -----------
    analysis_df : pandas.DataFrame
        Датафрейм с результатами анализа признаков, содержащий колонки:
        - feature: названия признаков
        - pearson_corr: коэффициенты корреляции Пирсона
        - spearman_corr: коэффициенты корреляции Спирмена  
        - skewness: значения асимметрии
        - kurtosis: значения эксцесса
    
    Returns:
    --------
    None
        Функция отображает графики и не возвращает значений
    """
    
    plt.figure(figsize=(15, 6))
    
    # График 1: Сравнение корреляций
    plt.subplot(1, 2, 1)
    indices = range(len(analysis_df))
    width = 0.35
    plt.bar([i - width/2 for i in indices], analysis_df['pearson_corr'], 
            width, label='Pearson', alpha=0.7)
    plt.bar([i + width/2 for i in indices], analysis_df['spearman_corr'], 
            width, label='Spearman', alpha=0.7)
    plt.xticks(indices, analysis_df['feature'], rotation=90, ha='right')
    plt.legend()
    plt.title('Сравнение корреляций Пирсона и Спирмена')
    plt.grid(True, alpha=0.3)
    
    # График 2: Распределение признаков (skewness + kurtosis)
    plt.subplot(1, 2, 2)
    plt.scatter(analysis_df['skewness'], analysis_df['kurtosis'], 
               s=100, alpha=0.7)
    for i, row in analysis_df.iterrows():
        plt.annotate(row['feature'], (row['skewness'], row['kurtosis']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('Skewness')
    plt.ylabel('Kurtosis')
    plt.title('Распределение признаков (skewness vs kurtosis)')
    plt.grid(True, alpha=0.3)    
    
    plt.tight_layout()
    plt.show()


def plot_feature_trends(df: pd.DataFrame,
                        metrics_df: pd.DataFrame,
                        features: list,
                        target: str,
                        config: Dict[str, Any],
                        figsize: Tuple[int, int] = (16, 12)) -> None:
    """
    Строит scatter plot с трендами для каждого признака.
    """
    def linear_func(x, a, b):
        return a * x + b
    
    def log_func(x, a, b):
        return a * np.log(x + 1e-10) + b
    
    def sqrt_func(x, a, b):
        return a * np.sqrt(x) + b
    
    def reciprocal_func(x, a, b):
        return a / (x + 1e-10) + b
    
    def square_func(x, a, b):
        return a * (x ** 2) + b
    
    n_features = len(features)
    n_cols = 2
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    colors = config['trend_settings']['colors']
    trend_names = config['trend_settings']['names']
    trend_funcs = [linear_func, log_func, sqrt_func, reciprocal_func, square_func]
    
    for i, feature in enumerate(features):
        if i >= len(axes):
            break
            
        ax = axes[i]
        x_data = df[feature].values
        y_data = df[target].values
        
        ax.scatter(x_data, y_data, alpha=0.6, s=30, color='gray', label='Data')
        ax.set_xlabel(feature)
        ax.set_ylabel(target)
        
        sort_idx = np.argsort(x_data)
        x_sorted = x_data[sort_idx]
        y_sorted = y_data[sort_idx]
        
        feature_metrics = metrics_df[metrics_df['feature'] == feature].iloc[0]
        
        for func, color, name in zip(trend_funcs, colors, trend_names):
            try:
                popt, _ = curve_fit(func, x_sorted, y_sorted, maxfev=5000)
                y_trend = func(x_sorted, *popt)
                r2 = feature_metrics[f'{name.lower().replace("²", "").replace("/", "")}_r2_score']
                
                ax.plot(x_sorted, y_trend, color=color, linewidth=2,
                       label=f'{name} (R²={r2:.3f})', alpha=0.8)
            except Exception:
                continue
        
        best_trend = feature_metrics['best_transformation']
        best_r2 = feature_metrics['best_r2_score']
        
        pearson_corr = np.corrcoef(x_data, y_data)[0, 1]
        spearman_corr = pd.Series(x_data).corr(pd.Series(y_data), method='spearman')
        
        ax.set_title(f'{feature} vs {target}\nPearson: {pearson_corr:.3f}, Spearman: {spearman_corr:.3f}')
        ax.legend(loc='best', fontsize=8)
        ax.annotate(f'Best: {best_trend} (R²={best_r2:.3f})',
                   xy=(0.05, 0.95), xycoords='axes fraction',
                   fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()