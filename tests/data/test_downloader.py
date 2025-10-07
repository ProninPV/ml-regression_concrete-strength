import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock, ANY
import sys
import subprocess

# Добавляем путь к src для корректного импорта
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from src.data.downloader import download_and_extract_competition, _has_extracted_files


def test_download_success():
    """Тест успешной загрузки - самый важный сценарий"""
    with patch('src.data.downloader.subprocess.check_call') as mock_kaggle:
        with patch('src.data.downloader._has_extracted_files') as mock_has_files:
            # Симулируем, что файлов еще нет
            mock_has_files.return_value = False
            
            # Создаем конфиг для теста
            config = {
                "competition": {
                    "name": "test-competition"
                }
            }
            
            # Вызываем функцию
            download_and_extract_competition(config)
            
            # Проверяем, что kaggle был вызван с правильными параметрами
            mock_kaggle.assert_called_once_with([
                "kaggle", "competitions", "download", "-c",
                "test-competition", "-p", ANY  # Используем ANY из unittest.mock
            ])
            
            # Проверяем, что в пути есть 'data/raw'
            call_args = mock_kaggle.call_args[0][0]
            download_path = call_args[6]  # Путь находится на 7-й позиции
            assert "data/raw" in download_path.replace("\\", "/")


def test_creates_missing_directory():
    """Тест создания отсутствующей директории"""
    with patch('src.data.downloader.subprocess.check_call'):
        with patch('src.data.downloader.os.makedirs') as mock_makedirs:
            with patch('src.data.downloader._has_extracted_files') as mock_has_files:
                mock_has_files.return_value = False
                
                config = {
                    "competition": {
                        "name": "test-competition"
                    }
                }
                
                # Вызываем функцию
                download_and_extract_competition(config)
                
                # Проверяем, что makedirs был вызван
                mock_makedirs.assert_called_once()


def test_handles_kaggle_not_found():
    """Тест обработки отсутствия kaggle CLI"""
    with patch('src.data.downloader.subprocess.check_call') as mock_kaggle:
        with patch('src.data.downloader._has_extracted_files') as mock_has_files:
            mock_has_files.return_value = False
            mock_kaggle.side_effect = FileNotFoundError("kaggle not found")
            
            config = {
                "competition": {
                    "name": "test-competition"
                }
            }
            
            # Функция не должна падать с ошибкой
            download_and_extract_competition(config)


def test_force_redownload_flag():
    """Тест работы флага force_redownload"""
    with patch('src.data.downloader.subprocess.check_call') as mock_kaggle:
        with patch('src.data.downloader._has_extracted_files') as mock_has_files:
            # Симулируем, что файлы уже существуют
            mock_has_files.return_value = True
            
            config = {
                "competition": {
                    "name": "test-competition"
                }
            }
            
            # Без force_redownload - не должно скачивать
            download_and_extract_competition(config, force_redownload=False)
            mock_kaggle.assert_not_called()
            
            # С force_redownload - должно скачать (несмотря на существующие файлы)
            download_and_extract_competition(config, force_redownload=True)
            mock_kaggle.assert_called_once()


def test_competition_name_from_config():
    """Тест получения имени конкурса из конфига"""
    with patch('src.data.downloader.subprocess.check_call') as mock_kaggle:
        with patch('src.data.downloader._has_extracted_files') as mock_has_files:
            mock_has_files.return_value = False
            
            config = {
                "competition": {
                    "name": "different-competition"
                }
            }
            
            download_and_extract_competition(config)
            
            # Проверяем, что использовалось имя из конфига
            call_args = mock_kaggle.call_args[0][0]
            competition_name_in_call = call_args[4]  # Имя конкурса на 5-й позиции
            assert competition_name_in_call == "different-competition"


def test_has_extracted_files():
    """Тест функции проверки существующих файлов"""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Создаем тестовые файлы
        with open(os.path.join(temp_dir, "test.csv"), "w") as f:
            f.write("test")
        
        # Должен вернуть True, так как есть не-ZIP файл
        assert _has_extracted_files(temp_dir) == True
        
        # Удаляем не-ZIP файл, создаем ZIP
        os.remove(os.path.join(temp_dir, "test.csv"))
        with open(os.path.join(temp_dir, "test.zip"), "w") as f:
            f.write("test")
        
        # Должен вернуть False, так как только ZIP файлы
        assert _has_extracted_files(temp_dir) == False


def test_subprocess_error_handling():
    """Тест обработки ошибок subprocess"""
    with patch('src.data.downloader.subprocess.check_call') as mock_kaggle:
        with patch('src.data.downloader._has_extracted_files') as mock_has_files:
            mock_has_files.return_value = False
            # Используем конкретное исключение, которое обрабатывается в функции
            mock_kaggle.side_effect = subprocess.CalledProcessError(1, "cmd")
            
            config = {
                "competition": {
                    "name": "test-competition"
                }
            }
            
            # Функция должна обработать ошибку и не упасть
            try:
                download_and_extract_competition(config)
                assert True  # Если дошли сюда - тест пройден
            except Exception:
                pytest.fail("Функция не обработала исключение из subprocess")


def test_general_exception_handling():
    """Тест обработки общих исключений при распаковке"""
    with patch('src.data.downloader.subprocess.check_call'):
        with patch('src.data.downloader._has_extracted_files') as mock_has_files:
            with patch('src.data.downloader.os.listdir') as mock_listdir:
                with patch('src.data.downloader.zipfile.ZipFile') as mock_zip:
                    mock_has_files.return_value = False
                    mock_listdir.return_value = ["test.zip"]
                    mock_zip.side_effect = Exception("Some zip error")
                    
                    config = {
                        "competition": {
                            "name": "test-competition"
                        }
                    }
                    
                    # Функция должна обработать ошибку и не упасть
                    try:
                        download_and_extract_competition(config)
                        assert True  # Если дошли сюда - тест пройден
                    except Exception:
                        pytest.fail("Функция не обработала исключение при распаковке")