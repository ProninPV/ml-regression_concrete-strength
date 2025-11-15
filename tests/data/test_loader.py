import os
import sys
import pytest
import pandas as pd
import tempfile
import yaml
import subprocess
from unittest.mock import patch, MagicMock
import pickle
from pathlib import Path

# Добавляем путь к src для импорта модулей
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.data.loader import load_config, install_package, data_load, data_load_preprocessed


class TestLoadConfig:
    """Тесты для функции load_config"""
    
    def test_load_config_with_valid_path(self):
        """Тест загрузки корректного конфиг файла"""
        # Создаем временный конфиг файл
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_content = """
            data:
                raw_dir: "data/raw"
                processed_dir: "data/processed"
                train_file: "train.csv"
                test_file: "test.csv"
            processed_files:
                train_file:
                    pkl: "train_processed.pkl"
                test_file:
                    pkl: "test_processed.pkl"
            model:
                name: "test_model"
            """
            f.write(config_content)
            config_path = f.name
        
        try:
            # Загружаем конфиг
            config = load_config(config_path)
            
            # Проверяем структуру
            assert "data" in config
            assert "model" in config
            assert "processed_files" in config
            assert config["data"]["raw_dir"] == "data/raw"
            assert config["data"]["processed_dir"] == "data/processed"
            assert config["data"]["train_file"] == "train.csv"
            assert config["processed_files"]["train_file"]["pkl"] == "train_processed.pkl"
            assert config["model"]["name"] == "test_model"
        finally:
            # Удаляем временный файл
            os.unlink(config_path)
    
    def test_load_config_with_none_path(self):
        """Тест загрузки конфига по умолчанию (путь=None)"""
        with patch('src.data.loader.os.path.exists') as mock_exists, \
             patch('src.data.loader.open') as mock_open, \
             patch('yaml.safe_load') as mock_load:
            
            # Настраиваем моки
            mock_exists.return_value = True
            mock_load.return_value = {"test": "config"}
            
            # Вызываем функцию
            config = load_config(None)
            
            # Проверяем вызовы
            assert config == {"test": "config"}
            mock_exists.assert_called_once()
            mock_open.assert_called_once()
    
    def test_load_config_file_not_found(self):
        """Тест обработки отсутствующего конфиг файла"""
        with tempfile.NamedTemporaryFile(suffix='.yaml') as f:
            # Файл будет автоматически удален после закрытия
            config_path = f.name
        
        # Проверяем, что возникает исключение
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config(config_path)
    
    def test_load_config_invalid_yaml(self):
        """Тест обработки некорректного YAML"""
        # Создаем временный файл с некорректным YAML
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_path = f.name
        
        try:
            # Проверяем, что возникает исключение
            with pytest.raises(ValueError, match="Error parsing YAML config"):
                load_config(config_path)
        finally:
            os.unlink(config_path)


class TestInstallPackage:
    """Тесты для функции install_package"""
    
    @patch('src.data.loader.importlib.import_module')
    @patch('src.data.loader.subprocess.check_call')
    def test_install_package_already_installed(self, mock_check_call, mock_import_module):
        """Тест случая, когда пакет уже установлен"""
        # Пакет успешно импортируется - ошибки нет
        mock_import_module.return_value = None
        
        install_package("existing-package")
        
        # Проверяем, что установка не вызывалась
        mock_check_call.assert_not_called()
    
    @patch('src.data.loader.importlib.import_module')
    @patch('src.data.loader.subprocess.check_call')
    def test_install_package_installation_success(self, mock_check_call, mock_import_module):
        """Тест успешной установки пакета"""
        # Имитируем ImportError при первой попытке импорта
        mock_import_module.side_effect = ImportError
        
        install_package("missing-package")
        
        # Проверяем, что установка была вызвана
        mock_check_call.assert_called_once_with([sys.executable, "-m", "pip", "install", "missing-package"])
    
    @patch('src.data.loader.importlib.import_module')
    @patch('src.data.loader.subprocess.check_call')
    def test_install_package_installation_failure(self, mock_check_call, mock_import_module):
        """Тест неудачной установки пакета"""
        # Имитируем ImportError при импорте
        mock_import_module.side_effect = ImportError
        # Имитируем ошибку при установке
        mock_check_call.side_effect = subprocess.CalledProcessError(1, "pip install")
        
        # Функция должна завершиться без исключения, но с сообщением об ошибке
        install_package("failing-package")
        
        # Проверяем, что установка была вызвана
        mock_check_call.assert_called_once_with([sys.executable, "-m", "pip", "install", "failing-package"])


class TestDataLoad:
    """Тесты для функции data_load"""
    
    @patch('pandas.read_csv')
    def test_data_load_train(self, mock_read_csv):
        """Тест загрузки тренировочных данных"""
        # Настраиваем мок для pandas.read_csv
        test_data = pd.DataFrame({
            'col1': [1, 4, 7],
            'col2': [2, 5, 8],
            'col3': [3, 6, 9]
        })
        mock_read_csv.return_value = test_data
        
        config = {
            "data": {
                "raw_dir": "data/raw",
                "train_file": "train.csv",
                "test_file": "test.csv"
            }
        }
        
        # Загружаем данные
        df = data_load('train', config)
        
        # Проверяем результат
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 3)
        assert list(df.columns) == ['col1', 'col2', 'col3']
        
        # Проверяем, что read_csv был вызван с правильным путем
        mock_read_csv.assert_called_once()
        actual_path = mock_read_csv.call_args[0][0]
        # Проверяем что путь содержит ожидаемые части (учитываем оба варианта разделителей)
        path_str = str(actual_path).replace('\\', '/')  # Нормализуем разделители
        assert 'data/raw' in path_str
        assert 'train.csv' in path_str
    
    @patch('pandas.read_csv')
    def test_data_load_test(self, mock_read_csv):
        """Тест загрузки тестовых данных"""
        # Настраиваем мок для pandas.read_csv
        test_data = pd.DataFrame({
            'feature1': [10, 30],
            'feature2': [20, 40],
            'target': [1, 0]
        })
        mock_read_csv.return_value = test_data
        
        config = {
            "data": {
                "raw_dir": "data/raw",
                "train_file": "train.csv",
                "test_file": "test.csv"
            }
        }
        
        # Загружаем данные
        df = data_load('test', config)
        
        # Проверяем результат
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 3)
        assert list(df.columns) == ['feature1', 'feature2', 'target']
        
        # Проверяем, что read_csv был вызван с правильным путем
        mock_read_csv.assert_called_once()
        actual_path = mock_read_csv.call_args[0][0]
        # Проверяем что путь содержит ожидаемые части (учитываем оба варианта разделителей)
        path_str = str(actual_path).replace('\\', '/')  # Нормализуем разделители
        assert 'data/raw' in path_str
        assert 'test.csv' in path_str
    
    def test_data_load_invalid_data_type(self):
        """Тест загрузки с некорректным типом данных"""
        config = {
            "data": {
                "raw_dir": "data/raw",
                "train_file": "train.csv",
                "test_file": "test.csv"
            }
        }
        
        # Проверяем, что возникает исключение
        with pytest.raises(ValueError, match="Неизвестный тип данных: invalid"):
            data_load('invalid', config)
    
    @patch('pandas.read_csv')
    def test_data_load_file_not_found(self, mock_read_csv):
        """Тест обработки отсутствующего файла"""
        # Настраиваем мок для выброса FileNotFoundError
        mock_read_csv.side_effect = FileNotFoundError("File not found")
        
        config = {
            "data": {
                "raw_dir": "data/raw",
                "train_file": "nonexistent.csv",
                "test_file": "test.csv"
            }
        }
        
        # Проверяем, что возникает исключение
        with pytest.raises(FileNotFoundError, match="Файл не найден"):
            data_load('train', config)
    
    @patch('pandas.read_csv')
    def test_data_load_general_exception(self, mock_read_csv):
        """Тест обработки общей ошибки при загрузке"""
        # Настраиваем мок для выброса общего исключения
        mock_read_csv.side_effect = Exception("General error")
        
        config = {
            "data": {
                "raw_dir": "data/raw",
                "train_file": "corrupted.csv",
                "test_file": "test.csv"
            }
        }
        
        # Проверяем, что возникает исключение
        with pytest.raises(Exception, match="Ошибка при загрузке данных"):
            data_load('train', config)


class TestDataLoadPreprocessed:
    """Тесты для функции data_load_preprocessed"""
    
    @patch('pandas.read_pickle')
    def test_data_load_preprocessed_train(self, mock_read_pickle):
        """Тест загрузки предобработанных тренировочных данных"""
        # Настраиваем мок для pandas.read_pickle
        test_data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'target': [0, 1, 0]
        })
        mock_read_pickle.return_value = test_data
        
        config = {
            "data": {
                "processed_dir": "data/processed"
            },
            "processed_files": {
                "train_file": {
                    "pkl": "train_processed.pkl"
                },
                "test_file": {
                    "pkl": "test_processed.pkl"
                }
            }
        }
        
        # Загружаем данные
        df = data_load_preprocessed('train', config)
        
        # Проверяем результат
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 3)
        assert list(df.columns) == ['feature1', 'feature2', 'target']
        
        # Проверяем, что read_pickle был вызван с правильным путем
        mock_read_pickle.assert_called_once()
        actual_path = mock_read_pickle.call_args[0][0]
        # Проверяем что путь содержит ожидаемые части (учитываем оба варианта разделителей)
        path_str = str(actual_path).replace('\\', '/')  # Нормализуем разделители
        assert 'data/processed' in path_str
        assert 'train_processed.pkl' in path_str
    
    @patch('pandas.read_pickle')
    def test_data_load_preprocessed_test(self, mock_read_pickle):
        """Тест загрузки предобработанных тестовых данных"""
        # Настраиваем мок для pandas.read_pickle
        test_data = pd.DataFrame({
            'feature1': [10, 20],
            'feature2': [30, 40],
            'feature3': [50, 60]
        })
        mock_read_pickle.return_value = test_data
        
        config = {
            "data": {
                "processed_dir": "data/processed"
            },
            "processed_files": {
                "train_file": {
                    "pkl": "train_processed.pkl"
                },
                "test_file": {
                    "pkl": "test_processed.pkl"
                }
            }
        }
        
        # Загружаем данные
        df = data_load_preprocessed('test', config)
        
        # Проверяем результат
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 3)
        assert list(df.columns) == ['feature1', 'feature2', 'feature3']
        
        # Проверяем, что read_pickle был вызван с правильным путем
        mock_read_pickle.assert_called_once()
        actual_path = mock_read_pickle.call_args[0][0]
        # Проверяем что путь содержит ожидаемые части (учитываем оба варианта разделителей)
        path_str = str(actual_path).replace('\\', '/')  # Нормализуем разделители
        assert 'data/processed' in path_str
        assert 'test_processed.pkl' in path_str
    
    def test_data_load_preprocessed_invalid_data_type(self):
        """Тест загрузки предобработанных данных с некорректным типом"""
        config = {
            "data": {
                "processed_dir": "data/processed"
            },
            "processed_files": {
                "train_file": {
                    "pkl": "train_processed.pkl"
                },
                "test_file": {
                    "pkl": "test_processed.pkl"
                }
            }
        }
        
        # Проверяем, что возникает исключение
        with pytest.raises(ValueError, match="Неизвестный тип данных: invalid"):
            data_load_preprocessed('invalid', config)
    
    @patch('pandas.read_pickle')
    def test_data_load_preprocessed_file_not_found(self, mock_read_pickle):
        """Тест обработки отсутствующего pickle файла"""
        # Настраиваем мок для выброса FileNotFoundError
        mock_read_pickle.side_effect = FileNotFoundError("File not found")
        
        config = {
            "data": {
                "processed_dir": "data/processed"
            },
            "processed_files": {
                "train_file": {
                    "pkl": "nonexistent.pkl"
                },
                "test_file": {
                    "pkl": "test_processed.pkl"
                }
            }
        }
        
        # Проверяем, что возникает исключение
        with pytest.raises(FileNotFoundError, match="Файл не найден"):
            data_load_preprocessed('train', config)
    
    @patch('pandas.read_pickle')
    def test_data_load_preprocessed_general_exception(self, mock_read_pickle):
        """Тест обработки общей ошибки при загрузке pickle"""
        # Настраиваем мок для выброса общего исключения
        mock_read_pickle.side_effect = Exception("Pickle error")
        
        config = {
            "data": {
                "processed_dir": "data/processed"
            },
            "processed_files": {
                "train_file": {
                    "pkl": "corrupted.pkl"
                },
                "test_file": {
                    "pkl": "test_processed.pkl"
                }
            }
        }
        
        # Проверяем, что возникает исключение
        with pytest.raises(Exception, match="Ошибка при загрузке данных"):
            data_load_preprocessed('train', config)


if __name__ == "__main__":
    # Запуск тестов при прямом выполнении скрипта
    pytest.main([__file__, "-v"])