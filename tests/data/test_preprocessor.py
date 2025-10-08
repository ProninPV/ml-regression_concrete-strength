import os
import sys
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import scipy.stats as stats

# Добавляем путь к src для импорта модулей
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.data.preprocessor import (
    add_concrete_ratios, 
    detect_outliers, 
    analyze_zeros, 
    preliminary_data_cleaning, 
    checking_order_of_features
)


class TestAddConcreteRatios:
    """Тесты для функции add_concrete_ratios"""
    
    def test_add_concrete_ratios_basic(self):
        """Тест добавления отношений с нормальными значениями"""
        df = pd.DataFrame({
            'Cement': [100, 200, 300],
            'Water': [50, 60, 90],
            'Superplasticizer': [2, 4, 6],
            'Other': [1, 2, 3]
        })
        
        result = add_concrete_ratios(df)
        
        # Проверяем добавление новых колонок
        assert 'W/C' in result.columns
        assert 'Sp/C_pct' in result.columns
        assert 'Other' in result.columns
        
        # Проверяем правильность вычислений
        expected_wc = [0.5, 0.3, 0.3]
        expected_spc = [0.02, 0.02, 0.02]
        
        np.testing.assert_array_almost_equal(result['W/C'].values, expected_wc)
        np.testing.assert_array_almost_equal(result['Sp/C_pct'].values, expected_spc)
    
    def test_add_concrete_ratios_with_zero_cement(self):
        """Тест обработки нулевых значений цемента"""
        df = pd.DataFrame({
            'Cement': [100, 0, 300],
            'Water': [50, 60, 90],
            'Superplasticizer': [2, 4, 6]
        })
        
        result = add_concrete_ratios(df)
        
        # Проверяем, что при Cement=0 значения становятся NaN
        assert pd.isna(result.loc[1, 'W/C'])
        assert pd.isna(result.loc[1, 'Sp/C_pct'])
        
        # Проверяем, что остальные значения вычислены правильно
        assert result.loc[0, 'W/C'] == 0.5
        assert result.loc[2, 'W/C'] == 0.3
    
    def test_add_concrete_ratios_original_df_unchanged(self):
        """Тест, что исходный DataFrame не изменяется"""
        df = pd.DataFrame({
            'Cement': [100, 200],
            'Water': [50, 60],
            'Superplasticizer': [2, 4]
        })
        df_original = df.copy()
        
        result = add_concrete_ratios(df)
        
        # Проверяем, что исходный DataFrame не изменился
        pd.testing.assert_frame_equal(df, df_original)
        # Проверяем, что результат - копия
        assert result is not df


class TestDetectOutliers:
    """Тесты для функции detect_outliers"""
    
    def test_detect_outliers_basic(self):
        """Тест базового обнаружения выбросов"""
        # Создаем данные с выбросами
        np.random.seed(42)
        normal_data = np.random.normal(100, 10, 90)
        outlier_data = np.array([50, 200])  # Явные выбросы
        data = np.concatenate([normal_data, outlier_data])
        
        df = pd.DataFrame({
            'feature1': data,
            'feature2': np.random.normal(50, 5, 92)  # Без выбросов
        })
        
        summary_df, outlier_masks_df = detect_outliers(df, config={})
        
        # Проверяем структуру результатов
        assert isinstance(summary_df, pd.DataFrame)
        assert isinstance(outlier_masks_df, pd.DataFrame)
        
        # Проверяем колонки в summary
        expected_columns = ['feature', 'Q1', 'Q3', 'IQR', 'Lower_Bound', 'Upper_Bound', 
                          'n_outliers_IQR', 'standard_min', 'standard_max', 'n_outliers_standard']
        assert all(col in summary_df.columns for col in expected_columns)
        
        # Проверяем, что выбросы обнаружены в feature1
        feature1_summary = summary_df[summary_df['feature'] == 'feature1'].iloc[0]
        assert feature1_summary['n_outliers_IQR'] == 3
    
    def test_detect_outliers_with_standard_comparison(self):
        """Тест сравнения со стандартными значениями"""
        df = pd.DataFrame({
            'cement': [100, 500, 1000],  # 500 и 1000 - выбросы по стандарту
            'water': [150, 200, 250]
        })
        
        config = {
            "standard_value": {
                "cement": {"min": 100, "max": 400},
                "water": {"min": 100, "max": 300}
            }
        }
        
        summary_df, _ = detect_outliers(df, config=config, compare_to_standard=True)
        
        cement_summary = summary_df[summary_df['feature'] == 'cement'].iloc[0]
        water_summary = summary_df[summary_df['feature'] == 'water'].iloc[0]
        
        # Проверяем стандартные значения
        assert cement_summary['standard_min'] == 100
        assert cement_summary['standard_max'] == 400
        assert cement_summary['n_outliers_standard'] == 2  # 500 и 1000
        
        assert water_summary['standard_min'] == 100
        assert water_summary['standard_max'] == 300
        assert water_summary['n_outliers_standard'] == 0  # Все в пределах
    
    def test_detect_outliers_without_standard_comparison(self):
        """Тест без сравнения со стандартными значениями"""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 100],  # 100 - выброс
            'feature2': [10, 11, 12, 13]
        })
        
        config = {"standard_value": {"feature1": {"min": 0, "max": 50}}}
        
        summary_df, _ = detect_outliers(df, config=config, compare_to_standard=False)
        
        feature1_summary = summary_df[summary_df['feature'] == 'feature1'].iloc[0]
        
        # Проверяем, что стандартные значения не заполнены
        assert feature1_summary['standard_min'] is None
        assert feature1_summary['standard_max'] is None
        assert feature1_summary['n_outliers_standard'] is None
    
    def test_detect_outliers_empty_config(self):
        """Тест с пустым конфигом"""
        df = pd.DataFrame({'col1': [1, 2, 3]})
        
        summary_df, outlier_masks_df = detect_outliers(df, config=None)
        
        # Проверяем, что функция работает без ошибок
        assert len(summary_df) == 1
        assert len(outlier_masks_df) == 3


class TestAnalyzeZeros:
    """Тесты для функции analyze_zeros"""
    
    @patch('scipy.stats.levene')
    @patch('scipy.stats.ttest_ind')
    def test_analyze_zeros_significant_difference(self, mock_ttest, mock_levene):
        """Тест случая со значимыми различиями"""
        # Настраиваем моки для статистических тестов
        mock_levene.return_value = (1.0, 0.5)  # p-value > 0.05 - равные дисперсии
        mock_ttest.return_value = (2.5, 0.01)  # p-value < 0.05 - значимое различие
        
        df = pd.DataFrame({
            'Blast Furnace Slag': [0, 0, 10, 20, 0, 15],
            'Fly Ash': [0, 5, 0, 10, 0, 0],
            'Superplasticizer': [0, 1, 2, 0, 3, 0],
            'Strength': [30, 35, 40, 45, 32, 38]
        })
        
        config = {}
        
        result_config = analyze_zeros(df, config)
        
        # Проверяем, что все признаки добавлены в конфиг (p-value < 0.05)
        assert 'binary_features' in result_config
        assert 'from_zeros' in result_config['binary_features']
        assert len(result_config['binary_features']['from_zeros']) == 3
        assert 'Blast Furnace Slag' in result_config['binary_features']['from_zeros']
        assert 'Fly Ash' in result_config['binary_features']['from_zeros']
        assert 'Superplasticizer' in result_config['binary_features']['from_zeros']
    
    @patch('scipy.stats.levene')
    @patch('scipy.stats.ttest_ind')
    def test_analyze_zeros_no_significant_difference(self, mock_ttest, mock_levene):
        """Тест случая без значимых различий"""
        mock_levene.return_value = (1.0, 0.5)
        mock_ttest.return_value = (1.0, 0.5)  # p-value > 0.05 - незначимое различие
        
        df = pd.DataFrame({
            'Blast Furnace Slag': [0, 0, 10, 20],
            'Fly Ash': [0, 5, 0, 10],
            'Superplasticizer': [0, 1, 2, 0],
            'Strength': [30, 35, 32, 33]  # Маленькая разница между группами
        })
        
        config = {}
        
        result_config = analyze_zeros(df, config)
        
        # Проверяем, что ни один признак не добавлен
        assert 'binary_features' in result_config
        assert len(result_config['binary_features']['from_zeros']) == 0
    
    @patch('scipy.stats.levene')
    @patch('scipy.stats.ttest_ind')
    def test_analyze_zeros_unequal_variances(self, mock_ttest, mock_levene):
        """Тест случая с неравными дисперсиями (Welch's t-test)"""
        mock_levene.return_value = (1.0, 0.01)  # p-value < 0.05 - неравные дисперсии
        mock_ttest.return_value = (2.5, 0.01)
        
        df = pd.DataFrame({
            'Blast Furnace Slag': [0, 0, 10, 20],
            'Fly Ash': [0, 5, 0, 10],
            'Superplasticizer': [0, 1, 2, 0],
            'Strength': [30, 35, 40, 45]
        })
        
        config = {}
        
        result_config = analyze_zeros(df, config)
        
        # Проверяем, что Welch's t-test был вызван (equal_var=False)
        mock_ttest.assert_called()
        call_args = mock_ttest.call_args
        assert call_args[1]['equal_var'] == False


class TestPreliminaryDataCleaning:
    """Тесты для функции preliminary_data_cleaning"""
    
    def test_preliminary_data_cleaning_basic(self):
        """Тест базовой очистки данных"""
        df = pd.DataFrame({
            'Id': [1, 2, 3, 4],
            'Cement': [100, 200, 300, 100],
            'Water': [50, 60, 70, 50],
            'Strength': [30, 35, 40, 30]  # Дубликат с первой строкой
        })
        
        result = preliminary_data_cleaning(df)
        
        # Проверяем удаление столбца Id
        assert 'Id' not in result.columns
        
        # Проверяем удаление дубликатов
        assert len(result) == 3  # Один дубликат удален
        
        # Проверяем сохранение остальных колонок
        assert 'Cement' in result.columns
        assert 'Water' in result.columns
        assert 'Strength' in result.columns
    
    def test_preliminary_data_cleaning_no_duplicates(self):
        """Тест очистки данных без дубликатов"""
        df = pd.DataFrame({
            'Id': [1, 2, 3],
            'Cement': [100, 200, 300],
            'Water': [50, 60, 70],
            'Strength': [30, 35, 40]
        })
        
        result = preliminary_data_cleaning(df)
        
        # Проверяем удаление столбца Id
        assert 'Id' not in result.columns
        
        # Проверяем, что все строки сохранены
        assert len(result) == 3
    
    def test_preliminary_data_cleaning_no_id_column(self):
        """Тест очистки данных без столбца Id"""
        df = pd.DataFrame({
            'Cement': [100, 200, 300],
            'Water': [50, 60, 70],
            'Strength': [30, 35, 40]
        })
        
        # Функция должна работать без ошибок
        result = preliminary_data_cleaning(df)
        
        # Проверяем, что данные не изменились
        assert len(result) == 3
        assert 'Cement' in result.columns
        assert 'Water' in result.columns
        assert 'Strength' in result.columns


class TestCheckingOrderOfFeatures:
    """Тесты для функции checking_order_of_features"""
    
    def test_checking_order_of_features_matching(self):
        """Тест совпадающего порядка признаков"""
        df_train = pd.DataFrame({
            'Cement': [100, 200, 300],
            'Water': [50, 60, 70],
            'Fly Ash': [10, 20, 30],
            'Strength': [30, 35, 40]
        })
        
        df_test = pd.DataFrame({
            'Cement': [400, 500],
            'Water': [80, 90],
            'Fly Ash': [40, 50]
        })
        
        result = checking_order_of_features(df_train, df_test)
        
        assert result == True
    
    def test_checking_order_of_features_different_order(self):
        """Тест разного порядка признаков"""
        df_train = pd.DataFrame({
            'Cement': [100, 200, 300],
            'Water': [50, 60, 70],
            'Fly Ash': [10, 20, 30],
            'Strength': [30, 35, 40]
        })
        
        df_test = pd.DataFrame({
            'Water': [80, 90],
            'Cement': [400, 500],
            'Fly Ash': [40, 50]
        })
        
        result = checking_order_of_features(df_train, df_test)
        
        assert result == False
    
    def test_checking_order_of_features_different_columns(self):
        """Тест разных наборов признаков"""
        df_train = pd.DataFrame({
            'Cement': [100, 200, 300],
            'Water': [50, 60, 70],
            'Fly Ash': [10, 20, 30],
            'Strength': [30, 35, 40]
        })
        
        df_test = pd.DataFrame({
            'Cement': [400, 500],
            'Water': [80, 90],
            'Superplasticizer': [5, 6]  # Другой признак
        })
        
        result = checking_order_of_features(df_train, df_test)
        
        assert result == False
    
    def test_checking_order_of_features_different_length(self):
        """Тест разного количества признаков"""
        df_train = pd.DataFrame({
            'Cement': [100, 200, 300],
            'Water': [50, 60, 70],
            'Fly Ash': [10, 20, 30],
            'Strength': [30, 35, 40]
        })
        
        df_test = pd.DataFrame({
            'Cement': [400, 500],
            'Water': [80, 90]
            # Отсутствует Fly Ash
        })
        
        result = checking_order_of_features(df_train, df_test)
        
        assert result == False


if __name__ == "__main__":
    # Запуск тестов при прямом выполнении скрипта
    pytest.main([__file__, "-v"])