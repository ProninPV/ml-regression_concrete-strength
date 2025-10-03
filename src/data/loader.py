import os
import sys
import subprocess
import yaml
import importlib
import pandas as pd


def load_config(config_path: str = None) -> dict:
    """
    Загружает YAML конфиг и возвращает dict.
    Если путь не указан, ищет config/config.yaml относительно корня проекта.
    """
    if config_path is None:
        # Корень проекта = три уровня выше (src/utils → src → project_root)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(project_root, "config", "config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML config: {e}")


def install_package(package_name: str) -> None:
    """
    Проверяет наличие Python-пакета и устанавливает его при отсутствии.

    Функция пытается импортировать указанный пакет. 
    Если пакет не найден (возникает ImportError), 
    выполняется его установка через pip.

    Args:
        package_name (str): Имя пакета, который требуется проверить/установить.

    Returns:
        None
    """
    try:
        importlib.import_module(package_name)
        print(f"[✓] Пакет {package_name} уже установлен")
    except ImportError:
        print(f"[⧗] Устанавливаю пакет {package_name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"[✓] Пакет {package_name} успешно установлен")
        except subprocess.CalledProcessError as e:
            print(f"[✗] Ошибка при установке пакета {package_name}: {e}")


def data_load(data_type: str, config: dict) -> pd.DataFrame:
    """
    Загружает данные из CSV файла.

    Parameters:
    -----------
    data_type : str
        Тип загружаемых данных. Допустимые значения:
        - 'train' - для загрузки обучающей выборки
        - 'test' - для загрузки тестовой выборки
    config : dict
        Словарь конфигурации

    Returns:
    --------
    pd.DataFrame
        DataFrame с загруженными данными.        
    """
    raw_dir = config["data"]["raw_dir"]
    
    if data_type == 'train':
        file_to_load = os.path.join('..', raw_dir, config["data"]["train_file"])
    elif data_type == 'test':
        file_to_load = os.path.join('..', raw_dir, config["data"]["test_file"])
    else:
        raise ValueError(f"Неизвестный тип данных: {data_type}. Допустимые значения: 'train', 'test'")
    
    print(f"[⧗] Загружаю данные из: {file_to_load}")
    
    try:
        df = pd.read_csv(file_to_load)
        print(f"[✓] Данные успешно загружены. Форма: {df.shape}")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл не найден: {file_to_load}")
    except Exception as e:
        raise Exception(f"Ошибка при загрузке данных: {e}")