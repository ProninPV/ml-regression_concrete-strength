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