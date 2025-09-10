import os
import yaml
import logging
import numpy as np
import importlib
import subprocess
import scipy.stats as stats
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Any, Optional, Tuple, Dict
from datetime import datetime


def load_config(config_path: str = None):
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

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


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
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])


def data_load(data_type: str, config: dict) -> pd.DataFrame:
    """
    data_type : str
        Тип загружаемых данных. Допустимые значения:
        - 'train' - для загрузки обучающей выборки
        - 'test' - для загрузки тестовой выборки
    config : dict, optional
        Словарь конфигурации

    Возвращает:
    -----------
    pd.DataFrame
        DataFrame с загруженными данными.        

    """
    
    raw_dir = config["data"]["raw_dir"]
    if data_type == 'train':
        file_to_load = os.path.join('..',raw_dir, config["data"]["train_file"])
    if data_type == 'test':
        file_to_load = os.path.join('..',raw_dir, config["data"]["test_file"])
    df = pd.read_csv(file_to_load)
    return df


def check_models_by_nepv(X: pd.DataFrame,
                         config: dict,
                         y: Optional[pd.Series] = None,                         
                         verbose: bool = True,
                         save_log: bool = True) -> List[str]:
    """
    Проверяет, какие модели соответствуют правилу NEPV (по Харреллу) для задачи регрессии или классификации.

    Параметры:
    ----------
    X : pd.DataFrame
        Признаковое пространство (только признаки, без целевой переменной).
    y : pd.Series
        Целевая переменная (для определения миноритарного класса при классификации).
    config : dict
        Конфигурация, содержащая как минимум ключи 'log_model_dir' и 'task_type'.
    verbose : bool, optional
        Если True, выводит результат в консоль.
    save_log : bool, optional
        Если True, сохраняет лог в файл в указанную директорию.

    Возвращает:
    -----------
    List[str]
        Список моделей, соответствующих правилу NEPV.
    """
    
    log_dir = os.path.join('..', config["data"]["logs"],"model_selection")
    task_type = config["task_type"]    
    
    n_features = X.shape[1]
    result_log = ""
    models = []

    if task_type == "regression":
        n_samples = X.shape[0]
        nepv = n_samples / n_features
        result_log += f"[REGRESSION] Правило NEPV: {n_samples} наблюдений / {n_features} признаков = {nepv:.2f}\n"

        if nepv >= 20:
            models += ["LinearRegression", "Ridge", "Lasso"]
            result_log += "LinearRegression / Ridge / Lasso — соответствуют (≥ 20)\n"
        else:
            result_log += "LinearRegression / Ridge / Lasso — не соответствуют (< 20)\n"

        if nepv >= 50:
            result_log += "CHAID (не реализован в sklearn) — соответствует (≥ 50)\n"
        else:
            result_log += "CHAID (не реализован в sklearn) — не соответствует (< 50)\n"

        if nepv >= 200:
            models += ["RandomForestRegressor", "GradientBoostingRegressor", "XGBRegressor", "LGBMRegressor"]
            result_log += "Сложные модели (RF, Boosting и др.) — соответствуют (≥ 200)\n"
        else:
            result_log += "Сложные модели (RF, Boosting и др.) — не соответствуют (< 200)\n"

        models += ["DecisionTreeRegressor"]
        result_log += "DecisionTreeRegressor (CART) — добавлен с осторожностью. NEPV к нему не применяется строго.\n"

    elif task_type == "classif":
        value_counts = y.value_counts()
        minority_class_count = value_counts.min()
        nepv = minority_class_count / n_features
        result_log += f"[CLASSIFICATION] NEPV: {minority_class_count} миноритарных / {n_features} признаков = {nepv:.2f}\n"

        if nepv >= 20:
            models += ["LogisticRegression"]
            result_log += "LogisticRegression — соответствует (≥ 20 событий на параметр)\n"
        else:
            result_log += "LogisticRegression — не соответствует (< 20)\n"

        if nepv >= 50:
            result_log += "CHAID (не реализован в sklearn) — соответствует (≥ 50 событий)\n"
        else:
            result_log += "CHAID (не реализован в sklearn) — не соответствует (< 50)\n"

        if nepv >= 200:
            models += ["RandomForestClassifier", "GradientBoostingClassifier", "XGBClassifier", "LGBMClassifier", "SVC", "MLPClassifier"]
            result_log += "Сложные модели (RF, Boosting, SVM, нейросети) — соответствуют (≥ 200)\n"
        else:
            result_log += "Сложные модели (RF, Boosting, SVM, нейросети) — не соответствуют (< 200)\n"

        models += ["DecisionTreeClassifier"]
        result_log += "DecisionTreeClassifier (CART) — добавлен с осторожностью. NEPV к нему не применяется строго.\n"

    else:
        raise ValueError(f"Неизвестный тип задачи: {task_type}")

    if verbose:
        print(result_log)

    if save_log:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_path = os.path.join(log_dir, f"nepv_check_{timestamp}.log")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(result_log)

    print('Список рекомендованных моделей:', models)
    return models


def eda_report(data: pd.DataFrame,
               dataset_name: str,
               config: dict,
               resave=True) -> None:
    """
    Генерирует EDA-отчёт с использованием ydata-profiling и сохраняет его в HTML-файл.
    Включает логирование, валидацию конфигурации, проверку на пропуски и предупреждения.

    Параметры:
    ----------
    data : pd.DataFrame
        Таблица с данными для анализа.

    dataset_name : str
        Название набора данных (будет использоваться в имени отчёта и заголовке).

    config_path : str
        Путь к YAML-файлу с конфигурацией. YAML должен содержать ключ 'eda_report_dir',
        указывающий путь для сохранения отчёта.

    Функциональность:
    -----------------    
    - Валидирует конфигурационный YAML-файл.
    - Создаёт каталог для отчётов, если он не существует.
    - Проверяет, существует ли HTML-отчёт — если да, не пересоздаёт.
    - Генерирует EDA-отчёт и сохраняет его в HTML-формате.
    - Выводит информацию о пропущенных значениях.
    - Извлекает и логирует предупреждения, выявленные ydata-profiling.
    - Ведёт логирование в файл и консоль.
    - Обрабатывает исключения с выводом полной информации.

    Возвращает:
    -----------
    None
        Функция не возвращает значение, но создаёт лог и отчёт в указанной директории.
    """

    from ydata_profiling import ProfileReport

    log_dir = os.path.join('..', config["data"]["logs"],"eda")
    log_path = os.path.join(log_dir, "eda_report.log")  # Полный путь к файлу
    
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("eda_report.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

    try:        

        output_dir = Path("..") / config["output"]["eda_report_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{dataset_name}_eda_report.html"
        
        if output_file.exists() and resave == False:
            logging.info(f"Отчёт уже существует: {output_file}. Пропускаем генерацию.")
            return
        
        # Генерация отчета
        logging.info(f"Генерация EDA отчета для '{dataset_name}'...")
        profile = ProfileReport(data, title=f"{dataset_name} EDA Report", explorative=True)
        profile.to_file(output_file)
        logging.info(f"Отчёт сохранён: {output_file}")

        # Проверка пропусков
        missing = data.isnull().sum()
        missing_total = missing.sum()
        if missing_total > 0:
            logging.warning(f"Обнаружены пропуски в {missing[missing > 0].shape[0]} колонках:")
            for col, count in missing[missing > 0].items():
                logging.warning(f" - {col}: {count} пропусков")
        else:
            logging.info("Пропусков в данных не обнаружено.")

        # Получение описания отчёта
        description = profile.get_description()

        # Получение списка предупреждений
        warnings = getattr(description, "warnings", [])

        if warnings:
            logging.warning("Предупреждения из отчета:")
            for warning in warnings:
                logging.warning(f" - {warning}")
        else:
            logging.info("Предупреждения ydata-profiling отсутствуют.")

    except Exception as e:
        logging.error(f"Ошибка при генерации EDA-отчета: {str(e)}", exc_info=True)


def plot_outliers(df: pd.DataFrame,
                  summary_df: pd.DataFrame,
                  max_plots: int = 20) -> None:
    """
    Строит boxplot-графики для признаков с наибольшим числом выбросов.

    max_plots — ограничение на количество отображаемых графиков
    """
    for _, row in summary_df.head(max_plots).iterrows():
        col = row['feature']
        plt.figure(figsize=(6, 1.5))
        sns.boxplot(x=df[col], color='skyblue')
        plt.title(f"Boxplot: {col} (IQR выбросов: {row['n_outliers_IQR']})")
        plt.tight_layout()
        plt.show()


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