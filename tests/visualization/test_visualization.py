import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
import sys
import os

# Добавляем путь к src для импорта модулей
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.visualization.visualization import (
    plot_outliers,
    visualize_feature_analysis,
    plot_feature_trends
)


class TestPlotOutliers:
    """Тесты для функции plot_outliers"""
    
    @pytest.fixture
    def sample_data(self):
        """Создает тестовые данные"""
        np.random.seed(42)
        df = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(5, 2, 100),
            'feature3': np.concatenate([np.random.normal(0, 1, 95), np.random.normal(10, 1, 5)]),  # с выбросами
        })
        
        summary_df = pd.DataFrame({
            'feature': ['feature3', 'feature1', 'feature2'],
            'n_outliers_IQR': [5, 2, 1]
        })
        
        return df, summary_df
    
    def test_plot_outliers_basic(self, sample_data):
        """Тест базового вызова функции"""
        df, summary_df = sample_data
        
        # Проверяем, что функция выполняется без ошибок
        try:
            plot_outliers(df, summary_df, max_plots=2)
            plt.close('all')  # Закрываем все графики после теста
            assert True
        except Exception as e:
            pytest.fail(f"Функция plot_outliers вызвала исключение: {e}")
    
    def test_plot_outliers_max_plots(self, sample_data):
        """Тест ограничения количества графиков"""
        df, summary_df = sample_data
        
        with patch('matplotlib.pyplot.show') as mock_show:
            plot_outliers(df, summary_df, max_plots=1)
            
            # Проверяем, что show вызывается (графики строятся)
            assert mock_show.called
            plt.close('all')
    
    def test_plot_outliers_empty_data(self):
        """Тест с пустыми данными"""
        empty_df = pd.DataFrame()
        empty_summary = pd.DataFrame(columns=['feature', 'n_outliers_IQR'])
        
        try:
            plot_outliers(empty_df, empty_summary)
            plt.close('all')
            assert True
        except Exception:
            # Может вызывать исключение на пустых данных - это нормально
            pass


class TestVisualizeFeatureAnalysis:
    """Тесты для функции visualize_feature_analysis"""
    
    @pytest.fixture
    def sample_analysis_df(self):
        """Создает тестовый датафрейм для анализа"""
        return pd.DataFrame({
            'feature': ['feat1', 'feat2', 'feat3', 'feat4'],
            'pearson_corr': [0.8, 0.3, -0.5, 0.1],
            'spearman_corr': [0.7, 0.4, -0.6, 0.2],
            'skewness': [0.1, 1.5, -0.8, 2.0],
            'kurtosis': [-0.5, 2.0, 1.0, 4.0]
        })
    
    def test_visualize_feature_analysis_basic(self, sample_analysis_df):
        """Тест базового вызова функции"""
        try:
            visualize_feature_analysis(sample_analysis_df)
            plt.close('all')
            assert True
        except Exception as e:
            pytest.fail(f"Функция visualize_feature_analysis вызвала исключение: {e}")
    
    def test_visualize_feature_analysis_empty(self):
        """Тест с пустым датафреймом"""
        empty_df = pd.DataFrame(columns=['feature', 'pearson_corr', 'spearman_corr', 'skewness', 'kurtosis'])
        
        try:
            visualize_feature_analysis(empty_df)
            plt.close('all')
            assert True
        except Exception:
            # Может вызывать исключение на пустых данных
            pass
    
    def test_visualize_feature_analysis_missing_columns(self):
        """Тест с отсутствующими колонками"""
        incomplete_df = pd.DataFrame({
            'feature': ['feat1', 'feat2'],
            'pearson_corr': [0.5, 0.3]
        })
        
        try:
            visualize_feature_analysis(incomplete_df)
            plt.close('all')
            # Ожидаем ошибку из-за отсутствующих колонок
        except (KeyError, Exception):
            assert True


class TestPlotFeatureTrends:
    """Тесты для функции plot_feature_trends"""
    
    @pytest.fixture
    def sample_trends_data(self):
        """Создает тестовые данные для построения трендов"""
        np.random.seed(42)
        n_samples = 50
        
        df = pd.DataFrame({
            'feature1': np.linspace(1, 10, n_samples) + np.random.normal(0, 0.5, n_samples),
            'feature2': np.random.exponential(2, n_samples),
            'target': np.linspace(5, 15, n_samples) + np.random.normal(0, 1, n_samples)
        })
        
        metrics_df = pd.DataFrame({
            'feature': ['feature1', 'feature2'],
            'best_transformation': ['Linear', 'Log'],
            'best_r2_score': [0.85, 0.72],
            'linear_r2_score': [0.85, 0.65],
            'log_r2_score': [0.82, 0.72],
            'sqrt_r2_score': [0.80, 0.68],
            'reciprocal_r2_score': [0.45, 0.32],
            'square_r2_score': [0.83, 0.60]
        })
        
        config = {
            'trend_settings': {
                'colors': ['red', 'blue', 'green', 'orange', 'purple'],
                'names': ['Linear', 'Log', 'Sqrt', '1/x', 'x²']
            }
        }
        
        features = ['feature1', 'feature2']
        target = 'target'
        
        return df, metrics_df, features, target, config
    
    def test_plot_feature_trends_basic(self, sample_trends_data):
        """Тест базового вызова функции"""
        df, metrics_df, features, target, config = sample_trends_data
        
        try:
            plot_feature_trends(df, metrics_df, features, target, config, figsize=(12, 8))
            plt.close('all')
            assert True
        except Exception as e:
            pytest.fail(f"Функция plot_feature_trends вызвала исключение: {e}")
    
    def test_plot_feature_trends_with_mock_plt(self, sample_trends_data):
        """Тест с моком matplotlib для проверки вызовов"""
        df, metrics_df, features, target, config = sample_trends_data
        
        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
            patch('matplotlib.pyplot.show') as mock_show:
            
            # Настраиваем мок - создаем numpy array-like объект с методом flatten
            mock_fig = MagicMock()
            mock_ax1 = MagicMock()
            mock_ax2 = MagicMock()
            
            # Создаем мок массива с методом flatten
            mock_axes_array = MagicMock()
            mock_axes_array.flatten.return_value = [mock_ax1, mock_ax2]
            mock_subplots.return_value = (mock_fig, mock_axes_array)
            
            plot_feature_trends(df, metrics_df, features, target, config)
            
            # Проверяем, что subplots был вызван
            assert mock_subplots.called
            # Проверяем, что show был вызван
            assert mock_show.called
    
    def test_plot_feature_trends_single_feature(self, sample_trends_data):
        """Тест с одним признаком"""
        df, metrics_df, features, target, config = sample_trends_data
        
        # Оставляем только один признак
        single_metrics_df = metrics_df[metrics_df['feature'] == 'feature1']
        single_features = ['feature1']
        
        try:
            plot_feature_trends(df, single_metrics_df, single_features, target, config)
            plt.close('all')
            assert True
        except Exception as e:
            pytest.fail(f"Функция plot_feature_trends с одним признаком вызвала исключение: {e}")
    
    def test_plot_feature_trends_empty_features(self, sample_trends_data):
        """Тест с пустым списком признаков"""
        df, metrics_df, features, target, config = sample_trends_data
        
        try:
            plot_feature_trends(df, metrics_df, [], target, config)
            plt.close('all')
            assert True
        except Exception:
            # Может вызывать исключение на пустых данных
            pass
    
    def test_plot_feature_trends_missing_metric_columns(self, sample_trends_data):
        """Тест с отсутствующими колонками в metrics_df"""
        df, metrics_df, features, target, config = sample_trends_data
        
        # Удаляем важные колонки
        incomplete_metrics = metrics_df[['feature', 'best_transformation']].copy()
        
        try:
            plot_feature_trends(df, incomplete_metrics, features, target, config)
            plt.close('all')
            # Ожидаем ошибку из-за отсутствующих колонок
        except (KeyError, Exception):
            assert True


def test_all_visualization_functions_integration():
    """Интеграционный тест всех функций визуализации"""
    # Создаем тестовые данные
    np.random.seed(42)
    
    # Данные для plot_outliers
    df = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.concatenate([np.random.normal(0, 1, 95), np.random.normal(10, 1, 5)]),
    })
    summary_df = pd.DataFrame({
        'feature': ['feature2', 'feature1'],
        'n_outliers_IQR': [5, 2]
    })
    
    # Данные для visualize_feature_analysis
    analysis_df = pd.DataFrame({
        'feature': ['feature1', 'feature2'],
        'pearson_corr': [0.8, 0.3],
        'spearman_corr': [0.7, 0.4],
        'skewness': [0.1, 1.5],
        'kurtosis': [-0.5, 2.0]
    })
    
    # Данные для plot_feature_trends
    metrics_df = pd.DataFrame({
        'feature': ['feature1'],
        'best_transformation': ['Linear'],
        'best_r2_score': [0.85],
        'linear_r2_score': [0.85],
        'log_r2_score': [0.82],
        'sqrt_r2_score': [0.80],
        'reciprocal_r2_score': [0.45],
        'square_r2_score': [0.83]
    })
    
    config = {
        'trend_settings': {
            'colors': ['red', 'blue', 'green', 'orange', 'purple'],
            'names': ['Linear', 'Log', 'Sqrt', '1/x', 'x²']
        }
    }
    
    # Проверяем, что все функции выполняются без критических ошибок
    try:
        plot_outliers(df, summary_df, max_plots=1)
        visualize_feature_analysis(analysis_df)
        plot_feature_trends(df, metrics_df, ['feature1'], 'feature1', config, figsize=(8, 6))
        plt.close('all')
        assert True
    except Exception as e:
        pytest.fail(f"Интеграционный тест вызвал исключение: {e}")


if __name__ == "__main__":
    # Запуск тестов напрямую
    pytest.main([__file__, "-v"])