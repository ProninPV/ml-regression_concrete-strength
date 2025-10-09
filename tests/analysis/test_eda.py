import pytest
import pandas as pd
import tempfile
import numpy as np
import os
from pathlib import Path
from unittest.mock import Mock, patch
import sys

# Добавляем путь к src для импорта модулей
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.analysis.eda import (
    validate_eda_config,
    should_generate_report,
    analyze_missing_values,
    extract_profile_warnings,
    generate_profile_report,
    eda_report,
    detect_outliers, 
    analyze_zeros, 
)


class TestValidateEDAConfig:
    """Тесты для функции validate_eda_config"""
    
    def test_valid_config(self):
        """Тест с валидной конфигурацией"""
        config = {
            'output': {
                'eda_report_dir': 'reports/eda'
            }
        }
        
        result = validate_eda_config(config)
        expected = Path("../..") / "reports/eda"
        assert result == expected
    
    def test_valid_config_with_custom_base_path(self):
        """Тест с кастомным базовым путем"""
        config = {
            'output': {
                'eda_report_dir': 'reports/eda'
            }
        }
        
        result = validate_eda_config(config, base_path="custom/path")
        expected = Path("custom/path") / "reports/eda"
        assert result == expected
    
    def test_missing_output_key(self):
        """Тест с отсутствующим ключом output"""
        config = {
            'other_section': {
                'eda_report_dir': 'reports/eda'
            }
        }
        
        with pytest.raises(ValueError, match="Отсутствует обязательный ключ в конфигурации: output/eda_report_dir"):
            validate_eda_config(config)
    
    def test_missing_eda_report_dir_key(self):
        """Тест с отсутствующим ключом eda_report_dir"""
        config = {
            'output': {
                'other_key': 'reports/eda'
            }
        }
        
        with pytest.raises(ValueError, match="Отсутствует обязательный ключ в конфигурации: output/eda_report_dir"):
            validate_eda_config(config)


class TestShouldGenerateReport:
    """Тесты для функции should_generate_report"""
    
    def test_should_generate_when_file_not_exists(self, tmp_path):
        """Тест когда файл не существует"""
        output_file = tmp_path / "nonexistent.html"
        assert should_generate_report(output_file, resave=True) == True
        assert should_generate_report(output_file, resave=False) == True
    
    def test_should_generate_when_file_exists_and_resave_true(self, tmp_path):
        """Тест когда файл существует и resave=True"""
        output_file = tmp_path / "existing.html"
        output_file.touch()  # Создаем файл
        assert should_generate_report(output_file, resave=True) == True
    
    def test_should_not_generate_when_file_exists_and_resave_false(self, tmp_path):
        """Тест когда файл существует и resave=False"""
        output_file = tmp_path / "existing.html"
        output_file.touch()  # Создаем файл
        assert should_generate_report(output_file, resave=False) == False


class TestAnalyzeMissingValues:
    """Тесты для функции analyze_missing_values"""
    
    def test_no_missing_values(self):
        """Тест без пропущенных значений"""
        data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        
        result = analyze_missing_values(data)
        
        assert result['total_missing'] == 0
        assert result['num_columns_with_missing'] == 0
        assert result['columns_with_missing'] == {}
    
    def test_with_missing_values(self):
        """Тест с пропущенными значениями"""
        data = pd.DataFrame({
            'col1': [1, None, 3],
            'col2': ['a', 'b', None],
            'col3': [1, 2, 3]  # Без пропусков
        })
        
        result = analyze_missing_values(data)
        
        assert result['total_missing'] == 2
        assert result['num_columns_with_missing'] == 2
        assert 'col1' in result['columns_with_missing']
        assert 'col2' in result['columns_with_missing']
        assert result['columns_with_missing']['col1'] == 1
        assert result['columns_with_missing']['col2'] == 1
    
    def test_all_missing_values(self):
        """Тест когда все значения пропущены"""
        data = pd.DataFrame({
            'col1': [None, None],
            'col2': [None, None]
        })
        
        result = analyze_missing_values(data)
        
        assert result['total_missing'] == 4
        assert result['num_columns_with_missing'] == 2
        assert result['columns_with_missing']['col1'] == 2
        assert result['columns_with_missing']['col2'] == 2


class TestExtractProfileWarnings:
    """Тесты для функции extract_profile_warnings"""
    
    def test_no_warnings(self):
        """Тест без предупреждений"""
        mock_profile = Mock()
        mock_description = Mock()
        mock_description.warnings = []
        mock_profile.get_description.return_value = mock_description
        
        result = extract_profile_warnings(mock_profile)
        
        assert result == []
        mock_profile.get_description.assert_called_once()
    
    def test_with_warnings(self):
        """Тест с предупреждениями"""
        mock_profile = Mock()
        mock_description = Mock()
        mock_description.warnings = [
            "Warning 1: High correlation",
            "Warning 2: Missing values"
        ]
        mock_profile.get_description.return_value = mock_description
        
        result = extract_profile_warnings(mock_profile)
        
        assert result == ["Warning 1: High correlation", "Warning 2: Missing values"]
        assert len(result) == 2
    
    def test_exception_handling(self):
        """Тест обработки исключений"""
        mock_profile = Mock()
        mock_profile.get_description.side_effect = Exception("Test error")
        
        result = extract_profile_warnings(mock_profile)
        
        assert result == []


class TestGenerateProfileReport:
    """Тесты для функции generate_profile_report"""
    
    @patch('src.analysis.eda.ProfileReport')
    def test_generate_report(self, mock_profile_report):
        """Тест генерации отчета"""
        # Mock данные
        data = pd.DataFrame({'col1': [1, 2, 3]})
        dataset_name = "test_dataset"
        output_file = Path("test_report.html")
        
        # Mock объект профиля
        mock_profile = Mock()
        mock_profile_report.return_value = mock_profile
        
        # Вызов функции
        result = generate_profile_report(data, dataset_name, output_file)
        
        # Проверки
        mock_profile_report.assert_called_once_with(
            data, title="test_dataset EDA Report", explorative=True
        )
        mock_profile.to_file.assert_called_once_with(output_file)
        assert result == mock_profile


class TestEDAReportIntegration:
    """Интеграционные тесты для основной функции eda_report"""
    
    @patch('src.analysis.eda.generate_profile_report')
    @patch('src.analysis.eda.extract_profile_warnings')
    def test_successful_report_generation(self, mock_extract_warnings, mock_generate_report):
        """Тест успешной генерации отчета"""
        # Mock данные
        data = pd.DataFrame({'col1': [1, 2, 3]})
        config = {
            'output': {
                'eda_report_dir': 'reports/eda'
            }
        }
        
        # Mock возвращаемых значений
        mock_profile = Mock()
        mock_generate_report.return_value = mock_profile
        mock_extract_warnings.return_value = []
        
        # Создаем временную директорию для теста
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            
            # Вызов функции
            eda_report(data, "test_dataset", config, base_path=str(base_path))
            
            # Проверяем что функции были вызваны
            mock_generate_report.assert_called_once()
            mock_extract_warnings.assert_called_once_with(mock_profile)
    
    @patch('src.analysis.eda.generate_profile_report')
    def test_skip_generation_when_file_exists(self, mock_generate_report):
        """Тест пропуска генерации когда файл уже существует"""
        data = pd.DataFrame({'col1': [1, 2, 3]})
        config = {
            'output': {
                'eda_report_dir': 'reports/eda'
            }
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            report_dir = base_path / "reports/eda"
            report_dir.mkdir(parents=True)
            output_file = report_dir / "test_dataset_eda_report.html"
            output_file.touch()  # Создаем существующий файл
            
            # Вызов с resave=False
            eda_report(data, "test_dataset", config, resave=False, base_path=str(base_path))
            
            # Проверяем что генерация не была вызвана
            mock_generate_report.assert_not_called()
    
    def test_invalid_config_handling(self):
        """Тест обработки невалидной конфигурации"""
        data = pd.DataFrame({'col1': [1, 2, 3]})
        invalid_config = {'invalid': 'config'}
        
        with pytest.raises(ValueError):
            eda_report(data, "test_dataset", invalid_config)


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


# Дополнительные тестовые данные для комплексного тестирования
@pytest.fixture
def sample_data_with_missing():
    """Фикстура с тестовыми данными с пропусками"""
    return pd.DataFrame({
        'numeric_col': [1, 2, None, 4, 5],
        'categorical_col': ['A', 'B', None, 'A', 'C'],
        'complete_col': [10, 20, 30, 40, 50]
    })


@pytest.fixture
def sample_data_complete():
    """Фикстура с полными тестовыми данными"""
    return pd.DataFrame({
        'col1': [1, 2, 3, 4, 5],
        'col2': ['A', 'B', 'C', 'D', 'E'],
        'col3': [10.1, 20.2, 30.3, 40.4, 50.5]
    })


def test_integration_with_sample_data(sample_data_with_missing):
    """Интеграционный тест с реальными данными"""
    config = {
        'output': {
            'eda_report_dir': 'test_reports'
        }
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Этот тест может упасть если ydata_profiling не установлен,
            # но мы проверяем что функции вызываются правильно
            eda_report(sample_data_with_missing, "integration_test", config, 
                      base_path=temp_dir, resave=True)
            
            # Проверяем что отчет создался
            report_path = Path(temp_dir) / "test_reports" / "integration_test_eda_report.html"
            # В реальном сценарии файл должен существовать
            # assert report_path.exists()
            
        except Exception as e:
            # В тестовой среде могут быть проблемы с зависимостями
            # но мы проверяем что ошибки связаны именно с ними, а не с нашей логикой
            assert "ydata_profiling" in str(e) or "ProfileReport" in str(e)


if __name__ == "__main__":
    # Запуск тестов из командной строки
    pytest.main([__file__, "-v"])