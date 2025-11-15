import pytest
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from datetime import datetime
from unittest.mock import patch, MagicMock
import logging

# Добавляем путь к исходному коду в Python path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data.saving import save_outliers_report, save_cleaned_data


class TestSaveOutliersReport:
    """Тесты для функции save_outliers_report"""
    
    @pytest.fixture
    def sample_summary_df(self):
        """Фикстура с примером DataFrame для тестирования"""
        return pd.DataFrame({
            'column': ['col1', 'col2', 'col3'],
            'outliers_count': [5, 3, 7],
            'outliers_percentage': [2.5, 1.5, 3.5],
            'method': ['IQR', 'IQR', 'Z-score']
        })
    
    @pytest.fixture
    def temp_output_dir(self):
        """Фикстура с временной директорией для тестов"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_save_outliers_report_creates_files(self, sample_summary_df, temp_output_dir):
        """Тест создания CSV и Excel файлов"""
        # Вызов функции
        save_outliers_report(sample_summary_df, temp_output_dir)
        
        # Проверка существования файлов
        csv_files = list(temp_output_dir.glob("*.csv"))
        xlsx_files = list(temp_output_dir.glob("*.xlsx"))
        
        assert len(csv_files) == 1, "Должен быть создан один CSV файл"
        assert len(xlsx_files) == 1, "Должен быть создан один Excel файл"
    
    def test_save_outliers_report_custom_prefix(self, sample_summary_df, temp_output_dir):
        """Тест с пользовательским префиксом имени файла"""
        custom_prefix = "custom_report"
        
        save_outliers_report(sample_summary_df, temp_output_dir, filename_prefix=custom_prefix)
        
        csv_files = list(temp_output_dir.glob(f"{custom_prefix}*.csv"))
        xlsx_files = list(temp_output_dir.glob(f"{custom_prefix}*.xlsx"))
        
        assert len(csv_files) == 1, "Должен быть создан CSV файл с кастомным префиксом"
        assert len(xlsx_files) == 1, "Должен быть создан Excel файл с кастомным префиксом"
    
    def test_save_outliers_report_creates_directory(self, sample_summary_df):
        """Тест создания несуществующей директории"""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_dir = Path(temp_dir) / "nonexistent" / "subdir"
            
            # Директория не должна существовать
            assert not nested_dir.exists()
            
            # Вызов функции должна создать директорию
            save_outliers_report(sample_summary_df, nested_dir)
            
            # Проверка что директория создана
            assert nested_dir.exists()
            assert nested_dir.is_dir()
    
    def test_save_outliers_report_file_content(self, sample_summary_df, temp_output_dir):
        """Тест содержимого сохраненных файлов"""
        save_outliers_report(sample_summary_df, temp_output_dir)
        
        # Проверка CSV файла
        csv_file = list(temp_output_dir.glob("*.csv"))[0]
        loaded_csv = pd.read_csv(csv_file)
        
        pd.testing.assert_frame_equal(sample_summary_df, loaded_csv)
        
        # Проверка Excel файла
        xlsx_file = list(temp_output_dir.glob("*.xlsx"))[0]
        loaded_xlsx = pd.read_excel(xlsx_file)
        
        pd.testing.assert_frame_equal(sample_summary_df, loaded_xlsx)
    
    @patch('builtins.print')
    def test_save_outliers_report_success_message(self, mock_print, sample_summary_df, temp_output_dir):
        """Тест вывода сообщения об успешном сохранении"""
        save_outliers_report(sample_summary_df, temp_output_dir)
        
        # Проверка что print был вызван с правильным сообщением
        mock_print.assert_called_once()
        call_args = mock_print.call_args[0][0]
        assert "Отчёт успешно сохранён" in call_args
    
    @patch('pandas.DataFrame.to_csv')
    def test_save_outliers_report_exception_handling(self, mock_to_csv, sample_summary_df, temp_output_dir):
        """Тест обработки исключений"""
        # Мокаем вызов to_csv чтобы вызвать исключение
        mock_to_csv.side_effect = Exception("Test error")
        
        # Перехватываем вывод
        with patch('builtins.print') as mock_print:
            save_outliers_report(sample_summary_df, temp_output_dir)
            
            # Проверка что выведено сообщение об ошибке
            mock_print.assert_called_once()
            call_args = mock_print.call_args[0][0]
            assert "Ошибка при сохранении отчёта" in call_args
            assert "Test error" in call_args


class TestSaveCleanedData:
    """Тесты для функции save_cleaned_data"""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Фикстура с примером DataFrame для очистки"""
        return pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10.5, 20.3, 30.1, 40.7, 50.9],
            'category': ['A', 'B', 'A', 'C', 'B']
        })
    
    @pytest.fixture
    def sample_config(self):
        """Фикстура с примером конфигурации"""
        return {
            'data': {
                'processed_dir': 'data/processed'
            }
        }
    
    def create_temp_project_structure(self):
        """Создает временную структуру проекта"""
        temp_dir = tempfile.mkdtemp()
        project_root = Path(temp_dir)
        (project_root / "src" / "data").mkdir(parents=True)
        (project_root / "data" / "processed").mkdir(parents=True)
        return project_root
    
    def cleanup_temp_project(self, temp_dir):
        """Очищает временную директорию"""
        shutil.rmtree(temp_dir)
    
    @patch('src.data.saving.logging')
    def test_save_cleaned_data_train_creates_files(self, mock_logging, sample_dataframe, sample_config):
        """Тест создания файлов для train данных"""
        temp_project_root = self.create_temp_project_structure()
        try:
            # Мокаем Path(__file__) чтобы вернуть правильный путь
            with patch('src.data.saving.Path') as mock_path_class:
                # Создаем мок для Path(__file__)
                mock_path_instance = MagicMock()
                mock_path_instance.resolve.return_value = temp_project_root / "src" / "data" / "saving.py"
                mock_path_instance.parent.parent.parent = temp_project_root
                
                # Настраиваем mock чтобы при вызове Path(__file__) возвращал наш мок
                mock_path_class.return_value = mock_path_instance
                
                # Вызов функции для train данных
                save_cleaned_data(sample_dataframe, 'train', sample_config)
            
            # Проверка что файлы созданы с правильными именами
            parquet_files = list((temp_project_root / "data" / "processed").glob("eda_data_train.parquet"))
            csv_files = list((temp_project_root / "data" / "processed").glob("eda_data_train.csv"))
            pkl_files = list((temp_project_root / "data" / "processed").glob("eda_data_train.pkl"))
            
            assert len(parquet_files) == 1, "Должен быть создан Parquet файл для train"
            assert len(csv_files) == 1, "Должен быть создан CSV файл для train"
            assert len(pkl_files) == 1, "Должен быть создан Pickle файл для train"
        finally:
            self.cleanup_temp_project(temp_project_root)
    
    @patch('src.data.saving.logging')
    def test_save_cleaned_data_test_creates_files(self, mock_logging, sample_dataframe, sample_config):
        """Тест создания файлов для test данных"""
        temp_project_root = self.create_temp_project_structure()
        try:
            # Мокаем Path(__file__) чтобы вернуть правильный путь
            with patch('src.data.saving.Path') as mock_path_class:
                # Создаем мок для Path(__file__)
                mock_path_instance = MagicMock()
                mock_path_instance.resolve.return_value = temp_project_root / "src" / "data" / "saving.py"
                mock_path_instance.parent.parent.parent = temp_project_root
                
                # Настраиваем mock чтобы при вызове Path(__file__) возвращал наш мок
                mock_path_class.return_value = mock_path_instance
                
                # Вызов функции для test данных
                save_cleaned_data(sample_dataframe, 'test', sample_config)
            
            # Проверка что файлы созданы с правильными именами
            parquet_files = list((temp_project_root / "data" / "processed").glob("eda_data_test.parquet"))
            csv_files = list((temp_project_root / "data" / "processed").glob("eda_data_test.csv"))
            pkl_files = list((temp_project_root / "data" / "processed").glob("eda_data_test.pkl"))
            
            assert len(parquet_files) == 1, "Должен быть создан Parquet файл для test"
            assert len(csv_files) == 1, "Должен быть создан CSV файл для test"
            assert len(pkl_files) == 1, "Должен быть создан Pickle файл для test"
        finally:
            self.cleanup_temp_project(temp_project_root)
    
    @patch('src.data.saving.logging')
    def test_save_cleaned_data_file_content(self, mock_logging, sample_dataframe, sample_config):
        """Тест содержимого сохраненных файлов"""
        temp_project_root = self.create_temp_project_structure()
        try:
            # Мокаем Path(__file__) чтобы вернуть правильный путь
            with patch('src.data.saving.Path') as mock_path_class:
                # Создаем мок для Path(__file__)
                mock_path_instance = MagicMock()
                mock_path_instance.resolve.return_value = temp_project_root / "src" / "data" / "saving.py"
                mock_path_instance.parent.parent.parent = temp_project_root
                
                # Настраиваем mock чтобы при вызове Path(__file__) возвращал наш мок
                mock_path_class.return_value = mock_path_instance
                
                # Вызов функции
                save_cleaned_data(sample_dataframe, 'train', sample_config)
            
            # Получаем пути к файлам
            parquet_file = temp_project_root / "data" / "processed" / "eda_data_train.parquet"
            csv_file = temp_project_root / "data" / "processed" / "eda_data_train.csv"
            pkl_file = temp_project_root / "data" / "processed" / "eda_data_train.pkl"
            
            # Проверка содержимого файлов
            # Parquet
            loaded_parquet = pd.read_parquet(parquet_file)
            pd.testing.assert_frame_equal(sample_dataframe, loaded_parquet)
            
            # CSV
            loaded_csv = pd.read_csv(csv_file)
            pd.testing.assert_frame_equal(sample_dataframe, loaded_csv)
            
            # Pickle
            loaded_pkl = pd.read_pickle(pkl_file)
            pd.testing.assert_frame_equal(sample_dataframe, loaded_pkl)
        finally:
            self.cleanup_temp_project(temp_project_root)
    
    @patch('src.data.saving.logging')
    def test_save_cleaned_data_creates_processed_dir(self, mock_logging, sample_dataframe, sample_config):
        """Тест создания директории processed если она не существует"""
        temp_project_root = self.create_temp_project_structure()
        try:
            # Удаляем директорию processed
            processed_dir = temp_project_root / "data" / "processed"
            shutil.rmtree(processed_dir)
            assert not processed_dir.exists()
            
            # Мокаем Path(__file__) чтобы вернуть правильный путь
            with patch('src.data.saving.Path') as mock_path_class:
                # Создаем мок для Path(__file__)
                mock_path_instance = MagicMock()
                mock_path_instance.resolve.return_value = temp_project_root / "src" / "data" / "saving.py"
                mock_path_instance.parent.parent.parent = temp_project_root
                
                # Настраиваем mock чтобы при вызове Path(__file__) возвращал наш мок
                mock_path_class.return_value = mock_path_instance
                
                # Вызов функции должна создать директорию
                save_cleaned_data(sample_dataframe, 'train', sample_config)
            
            # Проверка что директория создана
            assert processed_dir.exists()
            assert processed_dir.is_dir()
        finally:
            self.cleanup_temp_project(temp_project_root)
    
    @patch('src.data.saving.logging')
    def test_save_cleaned_data_logging(self, mock_logging, sample_dataframe, sample_config):
        """Тест логирования"""
        temp_project_root = self.create_temp_project_structure()
        try:
            # Мокаем Path(__file__) чтобы вернуть правильный путь
            with patch('src.data.saving.Path') as mock_path_class:
                # Создаем мок для Path(__file__)
                mock_path_instance = MagicMock()
                mock_path_instance.resolve.return_value = temp_project_root / "src" / "data" / "saving.py"
                mock_path_instance.parent.parent.parent = temp_project_root
                
                # Настраиваем mock чтобы при вызове Path(__file__) возвращал наш мок
                mock_path_class.return_value = mock_path_instance
                
                # Вызов функции
                save_cleaned_data(sample_dataframe, 'train', sample_config)
            
            # Проверка вызовов логирования
            assert mock_logging.info.called
            info_calls = [call[0][0] for call in mock_logging.info.call_args_list]
            
            # Проверка что были логи о сохранении файлов
            assert any("Данные успешно сохранены" in str(call) for call in info_calls)
            assert any("Parquet:" in str(call) for call in info_calls)
            assert any("CSV:" in str(call) for call in info_calls)
            assert any("Pickle:" in str(call) for call in info_calls)
        finally:
            self.cleanup_temp_project(temp_project_root)
    
    @patch('pandas.DataFrame.to_parquet')
    @patch('src.data.saving.logging')
    def test_save_cleaned_data_exception_handling(self, mock_logging, mock_to_parquet, 
                                                 sample_dataframe, sample_config):
        """Тест обработки исключений"""
        temp_project_root = self.create_temp_project_structure()
        try:
            # Мокаем вызов to_parquet чтобы вызвать исключение
            mock_to_parquet.side_effect = Exception("Parquet save error")
            
            # Мокаем Path(__file__) чтобы вернуть правильный путь
            with patch('src.data.saving.Path') as mock_path_class:
                # Создаем мок для Path(__file__)
                mock_path_instance = MagicMock()
                mock_path_instance.resolve.return_value = temp_project_root / "src" / "data" / "saving.py"
                mock_path_instance.parent.parent.parent = temp_project_root
                
                # Настраиваем mock чтобы при вызове Path(__file__) возвращал наш мок
                mock_path_class.return_value = mock_path_instance
                
                # Проверка что исключение пробрасывается
                with pytest.raises(Exception, match="Parquet save error"):
                    save_cleaned_data(sample_dataframe, 'train', sample_config)
            
            # Проверка логирования ошибки
            mock_logging.error.assert_called_once()
            error_call = mock_logging.error.call_args[0][0]
            assert "Ошибка при сохранении данных" in error_call
            assert "Parquet save error" in error_call
        finally:
            self.cleanup_temp_project(temp_project_root)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])