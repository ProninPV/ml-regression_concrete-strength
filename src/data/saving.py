import logging
from pathlib import Path
import pandas as pd
from datetime import datetime
from typing import List, Any, Optional, Tuple, Dict, Union

def save_outliers_report(summary_df: pd.DataFrame,   
                         output_dir: Path,
                         filename_prefix: str = "eda_outliers") -> None:
    """
    Сохраняет summary_df в .csv и .xlsx в указанную директорию.

    Параметры:
    ----------
    summary_df : pd.DataFrame
        Таблица с характеристиками выбросов.
    output_dir : Path
        Путь к директории, куда сохранять файлы.
    filename_prefix : str
        Префикс имени файла (по умолчанию 'eda_outliers').
    """
    try:      
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        now_str = datetime.now().strftime("%Y_%m_%d_%H-%M")  # заменил ':' на '-' для Windows
        filename_base = f"{filename_prefix}_{now_str}"

        csv_file = output_dir / f"{filename_base}.csv"
        
        xlsx_file = output_dir / f"{filename_base}.xlsx"

        summary_df.to_csv(csv_file, index=False)
        summary_df.to_excel(xlsx_file, index=False)

        print(f"Отчёт успешно сохранён:\n- {csv_file}\n- {xlsx_file}")

    except Exception as e:
        print(f"Ошибка при сохранении отчёта: {e}")


def save_cleaned_data(df: pd.DataFrame, data_type: str, config) -> dict:
    """
    Сохраняет очищенные данные в multiple formats (Parquet, CSV, Pickle)
    с добавлением даты в имя файла.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Очищенный DataFrame для сохранения
    config : dict
        Словарь с конфигурацией проекта, загруженный из config.yaml
        
    Returns:
    --------
    dict: Словарь с путями к сохраненным файлам
    """    
   
    # Получение пути для сохранения
    # Получаем корень проекта от текущей рабочей директории
    project_root = Path(__file__).resolve().parent.parent.parent

    base_path = project_root / config['data']['processed_dir']
    
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Создание имени файла
    if data_type == 'train':
        filename_base = f"eda_data_train"
    elif data_type == 'test':
        filename_base = f"eda_data_test"

    # Пути к файлам
    file_paths = {
        'parquet': base_path / f"{filename_base}.parquet",
        'csv': base_path / f"{filename_base}.csv", 
        'pkl': base_path / f"{filename_base}.pkl"
    }
    
    # Сохранение в разных форматах
    try:
        # 1. Parquet (основной формат)
        df.to_parquet(
            file_paths['parquet'], 
            index=False, 
            engine='pyarrow',
            compression='snappy'
        )
        
        # 2. CSV (для отладки)
        df.to_csv(
            file_paths['csv'],
            index=False,
            encoding='utf-8'
        )
        
        # 3. Pickle (для воспроизводимости)
        df.to_pickle(file_paths['pkl'])
        
        # Логирование
        logging.info(f"Данные успешно сохранены:")
        logging.info(f"Parquet: {file_paths['parquet']}")
        logging.info(f"CSV: {file_paths['csv']}")
        logging.info(f"Pickle: {file_paths['pkl']}")       
          
    except Exception as e:
        logging.error(f"Ошибка при сохранении данных: {e}")
        raise


def save_preprocessed_data(df: pd.DataFrame, config: Dict[str, Any]) -> dict:
    """
    Сохраняет данные после удаления выбросов по лучшей стратегии в multiple formats.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame после обработки выбросов для сохранения
    data_type : str
        Тип данных ('train', 'test' и т.д.)
    config : Dict[str, Any]
        Словарь с конфигурацией проекта
        
    Returns
    -------
    Dict[str, Path]
        Словарь с путями к сохраненным файлам:
        - 'parquet': путь к Parquet файлу
        - 'csv': путь к CSV файлу  
        - 'pkl': путь к Pickle файлу
        
    Raises
    ------
    Exception
        Если произошла ошибка при сохранении данных
    """ 
   
    # Получение пути для сохранения
    # Получаем корень проекта от текущей рабочей директории
    project_root = Path(__file__).resolve().parent.parent.parent

    base_path = project_root / config['data']['processed_dir']
    
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Создание имени файла    
    filename_base = f"data_train_outliers"
    
    # Пути к файлам
    file_paths = {
        'parquet': base_path / f"{filename_base}.parquet",
        'csv': base_path / f"{filename_base}.csv", 
        'pkl': base_path / f"{filename_base}.pkl"
    }
    
    # Сохранение в разных форматах
    try:
        # 1. Parquet (основной формат)
        df.to_parquet(
            file_paths['parquet'], 
            index=False, 
            engine='pyarrow',
            compression='snappy'
        )
        
        # 2. CSV (для отладки)
        df.to_csv(
            file_paths['csv'],
            index=False,
            encoding='utf-8'
        )
        
        # 3. Pickle (для воспроизводимости)
        df.to_pickle(file_paths['pkl'])
        
        # Логирование
        logging.info(f"Данные успешно сохранены:")
        logging.info(f"Parquet: {file_paths['parquet']}")
        logging.info(f"CSV: {file_paths['csv']}")
        logging.info(f"Pickle: {file_paths['pkl']}")       
          
    except Exception as e:
        logging.error(f"Ошибка при сохранении данных: {e}")
        raise


def save_preprocessed_target(y: pd.Series, config: Dict[str, Any]) -> dict:
    """
    Сохраняет таргет после удаления выбросов по лучшей стратегии в pkl.
    
    Parameters
    ----------
    y : pd.Series
        Series после обработки выбросов для сохранения    
    config : Dict[str, Any]
        Словарь с конфигурацией проекта
        
    Returns
    -------
    Dict[str, Path]
        - 'pkl': путь к Pickle файлу
        
    Raises
    ------
    Exception
        Если произошла ошибка при сохранении данных
    """ 
   
    # Получение пути для сохранения
    # Получаем корень проекта от текущей рабочей директории
    project_root = Path(__file__).resolve().parent.parent.parent

    base_path = project_root / config['data']['processed_dir']
    
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Создание имени файла    
    filename_base = f"y_train_outliers"
    
    # Пути к файлам
    file_paths = {
        'pkl': base_path / f"{filename_base}.pkl"
    }
    
    # Сохранение в разных форматах
    try:
        # Pickle (для воспроизводимости)
        y.to_pickle(file_paths['pkl'])
        
        # Логирование
        logging.info(f"Данные успешно сохранены:")
        logging.info(f"Pickle: {file_paths['pkl']}")       
          
    except Exception as e:
        logging.error(f"Ошибка при сохранении данных: {e}")
        raise