import os
import yaml
import logging
import pickle
import numpy as np
import scipy.stats as stats
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import levene
from scipy.stats import ttest_ind
from typing import List, Any, Optional, Tuple, Dict, Union
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import cross_val_score, KFold, RepeatedKFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import time
import psutil
from tqdm import tqdm
import gc

from ..features.feat_preprocessing import (OutlierHandler,
                                           FeatureHandleEngineering,
                                           FeatureTransformer,
                                           ZeroBinaryEncoder)

def check_models_by_nepv(X: pd.DataFrame,
                         config: dict,
                         y: Optional[pd.Series] = None,                         
                         verbose: bool = True,
                         save_log: bool = False) -> List[str]:
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
            models += ["RandomForestClassifier", "GradientBoostingClassifier",
                        "XGBClassifier", "LGBMClassifier", "SVC", "MLPClassifier"]
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


def create_full_pipeline(outlier_strategy: list[str],
                         model_type: str,
                         model: BaseEstimator,
                         feature_config,
                         y_name) -> Pipeline:
    """
    Создает полный пайплайн для предобработки данных и обучения модели.

    Пайплайн включает:
    1. Кастомный препроцессинг признаков
    2. Стандартизацию числовых признаков
    3. Модель машинного обучения

    Parameters
    ----------
    outlier_strategy : list[str]
        Список стратегий обработки выбросов
    model_type : str
        Тип модели ('regression' или 'classification')
    model : BaseEstimator
        Объект модели sklearn-совместимый

    Returns
    -------
    Pipeline
        Полный пайплайн для обучения и предсказаний
    """

    # Создаем pipeline для препроцессинга с помощью кастомных трансформеров
    preprocessor = Pipeline([
        ('feature_engineering', FeatureHandleEngineering(
            model_type=model_type,
            wc_column='W/C')),
        ('feature_transformer', FeatureTransformer(
            config=feature_config,
            target_col=y_name)),
        ('zero_binary_encoder', ZeroBinaryEncoder(alpha=0.05))
        # ('uninform_remove', feat_preprocessing.FeatureUninformRemove(threshold=0.95,
        #                                                              verbose=False))
        # ('collinearity_reducer', feat_preprocessing.CollinearityReducer())
    ])

    # Создаем полный пайплайн
    full_pipeline = Pipeline([
        ('preprocess', preprocessor),  # Было 'preprocess' вместо 'preprocessor'
        ('standard_scaler', StandardScaler()),  # Убрали лишние скобки
        ('model', model)  # Передаем модель напрямую, не оборачивая в Pipeline
    ])

    return full_pipeline


def run_experiments(X: pd.DataFrame,
                    y: pd.Series,
                    all_strategies: List[List[str]],  
                    model_types: List[str],
                    feature_config: Dict[str, Any],
                    models: Dict[str, BaseEstimator],                  
                    config: Dict[str, Any],
                    y_name: str) -> pd.DataFrame:
    """
    Проводит эксперименты по обучению моделей с различными стратегиями обработки выбросов.

    Для каждой комбинации стратегии выбросов, типа модели и конкретной модели:
    1. Обрабатывает выбросы в данных
    2. Создает и обучает пайплайн с препроцессингом и моделью  
    3. Оценивает качество с помощью повторяющейся кросс-валидации
    4. Измеряет время обучения и использование памяти

    Parameters
    ----------
    X : pd.DataFrame
        Матрица признаков для обучения
    y : pd.Series
        Вектор целевой переменной
    all_strategies : List[List[str]]
        Список стратегий обработки выбросов
    model_types : List[str]
        Список типов моделей для feature engineering
    feature_config : Dict[str, Any]
        Конфигурация для feature engineering
    models : Dict[str, BaseEstimator]
        Словарь моделей для тестирования
    config : Dict[str, Any]
        Основной конфигурационный словарь
    y_name : str
        Имя целевой переменной

    Returns
    -------
    pd.DataFrame
        DataFrame с результатами экспериментов, содержащий:
        - outlier_strategy: стратегия обработки выбросов
        - model_name: название модели
        - model_type: тип модели
        - dataset_size: размер данных после обработки выбросов
        - mean_rmse: среднее RMSE по кросс-валидации
        - std_rmse: стандартное отклонение RMSE
        - training_time_sec: время обучения в секундах
        - memory_used_mb: использование памяти в МБ
    """

    results = []

    total_combinations = len(all_strategies) * len(model_types) * len(models)

    with tqdm(total=total_combinations, desc="Running experiments", unit="model") as pbar:

        for outlier_strategy in all_strategies:
            for model_type in model_types:
                for model_name, model in models.items():

                    try:
                        gc.collect()
                        process = psutil.Process(os.getpid())
                        initial_memory = process.memory_info().rss / 1024 / 1024
                        start_time = time.time()

                        # 1. Предобработка выбросов ДО pipeline
                        outlier_handler = OutlierHandler(
                            strategies=outlier_strategy,
                            config=config,
                            target_col=y_name
                        )

                        X_processed, y_processed = outlier_handler.fit_transform(X, y)

                        # 2.ОБъявляем пайплайн с моделью
                        full_pipeline = create_full_pipeline(outlier_strategy,
                                                             model_type,
                                                             model,
                                                             feature_config,
                                                             y_name)

                        # 3. Кросс-валидация
                        kfold = RepeatedKFold(n_splits=6, n_repeats=5, random_state=42)

                        training_start = time.time()

                        rmse_scores = np.sqrt(-cross_val_score(
                            full_pipeline, X_processed, y_processed,
                            cv=kfold,
                            scoring='neg_mean_squared_error',
                            n_jobs=1
                        ))

                        training_time = time.time() - training_start

                        # Измерение памяти
                        final_memory = process.memory_info().rss / 1024 / 1024
                        memory_used = max(0, final_memory - initial_memory)

                        result = {
                            'outlier_strategy': str(outlier_strategy),
                            'model_name': model_name,
                            'model_type': model_type,
                            'dataset_size': X_processed.shape[0],
                            'mean_rmse': round(np.mean(rmse_scores), 4),
                            'std_rmse': round(np.std(rmse_scores), 4),
                            'training_time_sec': round(training_time, 2),
                            'memory_used_mb': round(memory_used, 2)
                        }
                        results.append(result)


                        pbar.set_postfix({
                            'model': model_name,
                            'time': f"{result['training_time_sec']}s",
                            'memory': f"{result['memory_used_mb']}MB"
                        })
                        pbar.update(1)

                    except Exception as e:
                        print(f"Error in {model_name} with {outlier_strategy}: {e}")
                        pbar.set_postfix({'model': model_name, 'status': 'ERROR'})
                        pbar.update(1)
                        continue

    return pd.DataFrame(results)


def get_best_model_strategy(modeling_result: pd.DataFrame) -> Dict[str, Any]:
    """
    Выбирает лучшую модель исходя из минимального RMSE.

    Parameters
    ----------
    modeling_result : pd.DataFrame
        DataFrame с результатами экспериментов

    Returns
    -------
    Dict[str, Any]
        Словарь с параметрами лучшей модели:
        - model: название лучшей модели
        - model_type: тип модели
        - outlier_strategy: стратегия обработки выбросов
        - rmse: значение RMSE
    """
    best_row = modeling_result.loc[modeling_result['mean_rmse'].idxmin()]

    return {
        'model': best_row['model_name'],
        'model_type': best_row['model_type'],
        'outlier_strategy': best_row['outlier_strategy'],
        'rmse': best_row['mean_rmse']
    }


def save_sorted_modeling_report(config: Dict[str, Any],
                                modeling_result: pd.DataFrame) -> None:
    """
    Сохраняет отсортированный отчет по моделированию.

    Parameters
    ----------
    config : Dict[str, Any]
        Конфигурационный словарь
    modeling_result : pd.DataFrame
        DataFrame с результатами экспериментов
    """
    # Сортируем по RMSE (по возрастанию)
    sorted_report = modeling_result.sort_values('mean_rmse')

    # Получаем путь и создаем директорию если не существует
    # report_path = config['output']['modeling_report_dir']
    # report_dir = os.path.dirname(report_path)
    project_root = Path(__file__).resolve().parent.parent.parent
    report_path = project_root / config['models_saving']['modeling_report_dir']
    report_path.mkdir(parents=True, exist_ok=True)

    # Генерируем имя файла с временем
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"modeling_experiments_{timestamp}.csv"

    report_path = report_path / filename 

    # Сохраняем в указанную директорию
    sorted_report.to_csv(report_path, index=False)
    print(f"Отчет сохранен: {report_path}")


def save_best_pipeline(best_model: str,
                       best_model_type: str,
                       best_outlier_strategy: str,
                       models: Dict[str, Any],
                       modeling_result: pd.DataFrame,
                       feature_config: Dict[str, Any],                       
                       config: Dict[str, Any],
                       y_name: str) -> None:
    """
    Создает и сохраняет пайплайн с лучшими параметрами и аннотацией.

    Parameters
    ----------
    best_model : str
        Название лучшей модели
    best_model_type : str
        Тип модели
    best_outlier_strategy : str
        Стратегия обработки выбросов
    models : Dict[str, Any]
        Словарь с объектами моделей
    modeling_result : pd.DataFrame
        DataFrame с результатами для извлечения полной информации

    """
    # Получаем объект модели
    model_obj = models[best_model]

    # Находим полную информацию о лучшей модели
    best_row = modeling_result.loc[modeling_result['mean_rmse'].idxmin()]

    # Создаем пайплайн с лучшими параметрами
    full_pipeline = create_full_pipeline(best_outlier_strategy,
                                         best_model_type,
                                         best_model,
                                         feature_config,
                                         y_name)

    # Создаем аннотацию с метаданными (без пайплайна - сохраняем отдельно)
    pipeline_metadata = {
        'metadata': {
            'best_model': best_model,
            'best_model_type': best_model_type,
            'best_outlier_strategy': best_outlier_strategy,
            'rmse': best_row['mean_rmse'],
            'rmse_std': best_row['std_rmse'],
            'dataset_size': best_row['dataset_size'],
            'training_time_sec': best_row['training_time_sec'],
            'memory_used_mb': best_row['memory_used_mb'],
            'created_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'experiment_info': 'Best pipeline from modeling experiments',
            'config': config  # Сохраняем конфиг для воспроизводимости
        }
    }
    
    # Получаем путь и создаем директорию если не существует
    # pipeline_dir = config['output']['best_pipeline_dir']  # Это должна быть директория, а не файл
    
    # Создаем директорию если не существует
    # os.makedirs(pipeline_dir, exist_ok=True)

    # Определяем путь сохранения pipeline
    project_root = Path(__file__).resolve().parent.parent.parent
    pipeline_dir = project_root / config['models_saving']['best_pipeline_dir']
    pipeline_dir.mkdir(parents=True, exist_ok=True)

    # Генерируем имя файла с аннотацией
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Сохраняем пайплайн и метаданные отдельно
    pipeline_name = f"best_pipeline_{best_model}.pkl"
    metadata_name = f"pipeline_metadata_{best_model}.pkl"

    pipeline_path = os.path.join(pipeline_dir, pipeline_name)
    metadata_path = os.path.join(pipeline_dir, metadata_name)

    try:
        # Пытаемся сохранить пайплайн
        with open(pipeline_path, 'wb') as f:
            pickle.dump(full_pipeline, f)
    except (pickle.PickleError, AttributeError) as e:
        print(f"Ошибка сохранения пайплайна: {e}")
        print("Создаем новый пайплайн для сохранения...")

        # Создаем новый пайплайн без проблемных трансформеров
        simple_pipeline = create_full_pipeline(
            outlier_strategy=best_outlier_strategy,
            model_type=best_model_type,
            model=model_obj
        )

        with open(pipeline_path, 'wb') as f:
            pickle.dump(simple_pipeline, f)

    # Сохраняем метаданные
    with open(metadata_path, 'wb') as f:
        pickle.dump(pipeline_metadata, f)

    print(f"Пайплайн сохранен: {pipeline_path}")
    print(f"Метаданные сохранены: {metadata_path}")
    print(f"Лучшая модель: {best_model}, RMSE: {best_row['mean_rmse']:.4f}")