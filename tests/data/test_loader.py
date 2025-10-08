import os
import sys
import pytest
import pandas as pd
import tempfile
import yaml
import subprocess  # Добавлен импорт subprocess
from unittest.mock import patch, MagicMock

# Добавляем путь к src для импорта модулей
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.data.loader import load_config, install_package, data_load


class TestLoadConfig:
    """Тесты для функции load_config"""
    
    def test_load_config_with_valid_path(self):
        """Тест загрузки корректного конфиг файла"""
        # Создаем временный конфиг файл
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_content = """
            data:
                raw_dir: "data/raw"
                train_file: "train.csv"
                test_file: "test.csv"
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
            assert config["data"]["raw_dir"] == "data/raw"
            assert config["data"]["train_file"] == "train.csv"
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
    
    def test_data_load_train(self):
        """Тест загрузки тренировочных данных"""
        # Создаем временный CSV файл
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df_content = """col1,col2,col3
1,2,3
4,5,6
7,8,9"""
            f.write(df_content)
            train_file_path = f.name
        
        try:
            # Создаем мок конфига
            config = {
                "data": {
                    "raw_dir": os.path.dirname(train_file_path),
                    "train_file": os.path.basename(train_file_path),
                    "test_file": "test.csv"
                }
            }
            
            # Загружаем данные
            df = data_load('train', config)
            
            # Проверяем результат
            assert isinstance(df, pd.DataFrame)
            assert df.shape == (3, 3)
            assert list(df.columns) == ['col1', 'col2', 'col3']
            assert df['col1'].tolist() == [1, 4, 7]
            
        finally:
            os.unlink(train_file_path)
    
    def test_data_load_test(self):
        """Тест загрузки тестовых данных"""
        # Создаем временный CSV файл
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df_content = """feature1,feature2,target
10,20,1
30,40,0"""
            f.write(df_content)
            test_file_path = f.name
        
        try:
            # Создаем мок конфига
            config = {
                "data": {
                    "raw_dir": os.path.dirname(test_file_path),
                    "train_file": "train.csv",
                    "test_file": os.path.basename(test_file_path)
                }
            }
            
            # Загружаем данные
            df = data_load('test', config)
            
            # Проверяем результат
            assert isinstance(df, pd.DataFrame)
            assert df.shape == (2, 3)
            assert list(df.columns) == ['feature1', 'feature2', 'target']
            
        finally:
            os.unlink(test_file_path)
    
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


if __name__ == "__main__":
    # Запуск тестов при прямом выполнении скрипта
    pytest.main([__file__, "-v"])