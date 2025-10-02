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
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr, kurtosis, skew
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


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


def create_feature_analysis(df: pd.DataFrame, target_column: pd.Series) -> pd.DataFrame:
    """
    Создает комплексный датафрейм для анализа признаков. Вычисляет коэффициенты Пирсона, Спирмена
    для оригинальных данных, а также значения асимметриb и эксцесса 
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Исходный датафрейм с признаками и целевой переменной
    target_column : pd.Series
        Целевая переменная
    
    Returns:
    --------
    pandas.DataFrame
        Датафрейм с метриками для каждого признака
    """
    
    # Исключаем ненужные колонки
    features = df.select_dtypes(include='number').columns.tolist()
    
    # Создаем список для хранения результатов
    results = []
    target_data = target_column
    
    for feature in features:
        feature_data = df[feature]
                
        # Базовые статистики
        pearson_corr, pearson_p = pearsonr(feature_data, target_data)
        spearman_corr, spearman_p = spearmanr(feature_data, target_data)
        skewness = skew(feature_data)
        kurtosis_val = kurtosis(feature_data)        
              
        # Собираем результаты
        results.append({
            'feature': feature,
            'pearson_corr': pearson_corr,
            'spearman_corr': spearman_corr,
            'spearman_p_value': round(spearman_p, 4),
            'skewness': skewness,
            'kurtosis': kurtosis_val            
        })
    
    # Создаем датафрейм
    analysis_df = pd.DataFrame(results)
    
    # Сортируем по абсолютной корреляции Спирмена
    analysis_df = analysis_df.sort_values('spearman_corr', ascending=False)
    
    return analysis_df.reset_index(drop=True)


def visualize_feature_analysis(analysis_df: pd.DataFrame) -> None:
    """
    Визуализирует сравнение корреляций Пирсона и Спирмена для каждого признака с целевой переменной и
    значения асимметрии и эксцесса для признаков.
    
    Parameters:
    -----------
    analysis_df : pandas.DataFrame
        Датафрейм с результатами анализа признаков, содержащий колонки:
        - feature: названия признаков
        - pearson_corr: коэффициенты корреляции Пирсона
        - spearman_corr: коэффициенты корреляции Спирмена  
        - skewness: значения асимметрии
        - kurtosis: значения эксцесса
    
    Returns:
    --------
    None
        Функция отображает графики и не возвращает значений
    """
    
    plt.figure(figsize=(15, 6))
    
    # График 1: Сравнение корреляций
    plt.subplot(1, 2, 1)
    indices = range(len(analysis_df))
    width = 0.35
    plt.bar([i - width/2 for i in indices], analysis_df['pearson_corr'], 
            width, label='Pearson', alpha=0.7)
    plt.bar([i + width/2 for i in indices], analysis_df['spearman_corr'], 
            width, label='Spearman', alpha=0.7)
    plt.xticks(indices, analysis_df['feature'], rotation=90, ha='right')
    plt.legend()
    plt.title('Сравнение корреляций Пирсона и Спирмена')
    plt.grid(True, alpha=0.3)
    
    # График 2: Распределение признаков (skewness + kurtosis)
    plt.subplot(1, 2, 2)
    plt.scatter(analysis_df['skewness'], analysis_df['kurtosis'], 
               s=100, alpha=0.7)
    for i, row in analysis_df.iterrows():
        plt.annotate(row['feature'], (row['skewness'], row['kurtosis']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('Skewness')
    plt.ylabel('Kurtosis')
    plt.title('Распределение признаков (skewness vs kurtosis)')
    plt.grid(True, alpha=0.3)    
    
    plt.tight_layout()
    plt.show()


def plot_feature_trends(df: pd.DataFrame,
                        config: Dict[str, Any],
                        target: str = 'Strength',
                        figsize: Tuple[int, int] = (16, 12),
                        alpha: float = 0.2) -> pd.DataFrame:
    """
    Строит scatter plot для каждого признака с наложением различных трендов.
    Возвращает датасет с лучшими преобразованиями и R² scores.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Исходный датафрейм с данными
    features : list
        Список признаков для анализа
    target : str
        Название целевой переменной
    figsize : tuple
        Размер figure для plotting
    alpha : float
        Пороговое значение для выбора линейного тренда (0.2 = 20%)
    """

    # Формирование списка признаков
    features = df.select_dtypes(include=[np.number]).columns
    features = [f for f in features if f != target]
    
    # Определяем функции для трендов
    def linear_func(x, a, b):
        return a * x + b
    
    def log_func(x, a, b):
        return a * np.log(x + 1e-10) + b
    
    def sqrt_func(x, a, b):
        return a * np.sqrt(x) + b
    
    def reciprocal_func(x, a, b):
        return a / (x + 1e-10) + b
    
    def square_func(x, a, b):
        return a * (x ** 2) + b
    
    # Создаем grid subplots
    n_features = len(features)
    n_cols = 2
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    # Цвета для разных трендов
    colors = config['trend_settings']['colors']
    trend_names = config['trend_settings']['names']
    trend_funcs = [linear_func, log_func, sqrt_func, reciprocal_func, square_func]
    
    # Создаем датасет для результатов
    results_data = []
    
    for i, feature in enumerate(features):
        if i >= len(axes):
            break
            
        ax = axes[i]
        x_data = df[feature].values
        y_data = df[target].values
        
        # Scatter plot
        ax.scatter(x_data, y_data, alpha=0.6, s=30, color='gray', label='Data')
        ax.set_xlabel(feature)
        ax.set_ylabel(target)
        ax.set_title(f'{feature} vs {target}')
        
        # Сортируем данные для построения трендов
        sort_idx = np.argsort(x_data)
        x_sorted = x_data[sort_idx]
        y_sorted = y_data[sort_idx]
        
        # Строим различные тренды
        r2_scores = {}
        
        for j, (func, color, name) in enumerate(zip(trend_funcs, colors, trend_names)):
            try:
                # Подбираем параметры для функции тренда
                popt, _ = curve_fit(func, x_sorted, y_sorted, maxfev=5000)
                
                # Предсказываем значения для тренда
                y_trend = func(x_sorted, *popt)
                
                # Вычисляем R² score
                r2 = r2_score(y_sorted, y_trend)
                r2_scores[name] = r2
                
                # Строим тренд
                ax.plot(x_sorted, y_trend, color=color, linewidth=2, 
                       label=f'{name} (R²={r2:.3f})', alpha=0.8)
                
            except Exception as e:
                # Пропускаем тренды, которые не удается построить
                r2_scores[name] = -np.inf
                continue
        
        # Определяем лучшее преобразование
        best_trend_name = max(r2_scores.items(), key=lambda x: x[1])[0]
        best_r2 = r2_scores[best_trend_name]
        linear_r2 = r2_scores.get('Linear', -np.inf)
        
        # Если линейный тренд хуже лучшего менее чем на alpha, выбираем линейный
        if best_trend_name != 'Linear' and linear_r2 >= best_r2 * (1 - alpha):
            final_best_trend = 'Linear'
            final_best_r2 = linear_r2
        else:
            final_best_trend = best_trend_name
            final_best_r2 = best_r2
        
        # Добавляем данные в результаты
        results_data.append({
            'feature': feature,
            'best_transformation': final_best_trend,
            'best_r2_score': final_best_r2,
            'linear_r2_score': linear_r2,
            'log_r2_score': r2_scores.get('Log', np.nan),
            'sqrt_r2_score': r2_scores.get('Sqrt', np.nan),
            'reciprocal_r2_score': r2_scores.get('1/x', np.nan),
            'square_r2_score': r2_scores.get('x²', np.nan)
        })
        
        # Добавляем легенду
        ax.legend(loc='best', fontsize=8)
        
        # Добавляем аннотацию с лучшим трендом
        ax.annotate(f'Best: {final_best_trend} (R²={final_best_r2:.3f})',
                   xy=(0.05, 0.95), xycoords='axes fraction',
                   fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        # Добавляем корреляции в заголовок
        pearson_corr = np.corrcoef(x_data, y_data)[0, 1]
        spearman_corr = pd.Series(x_data).corr(pd.Series(y_data), method='spearman')
        ax.set_title(f'{feature} vs {target}\nPearson: {pearson_corr:.3f}, Spearman: {spearman_corr:.3f}')
    
    # Скрываем пустые subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # Создаем и возвращаем датасет с результатами
    results_df = pd.DataFrame(results_data)
    return results_df


def check_multicollinearity(df: pd.DataFrame,
                            target_column: pd.Series, 
                            threshold: int = 10,
                            plot_heatmap: bool = True) -> pd.DataFrame:
    """
    Проверяет мультиколлинеарность для списка признаков в DataFrame.

    Parameters:
    df: Исходный DataFrame с данными.
    target_column: Целевая переменная.
    threshold: Пороговое значение VIF для выделения проблемных признаков. По умолчанию 10.
    plot_heatmap: Строить ли тепловую карту корреляций. По умолчанию True.

    Returns:
    pd.DataFrame: DataFrame с признаками и их VIF значениями, отсортированный по убыванию VIF.
    """

    features = df.select_dtypes(include='number').columns.tolist()
    target_data = target_column
    
    # Шаг 1: Матрица корреляций и тепловая карта
    if plot_heatmap:
        # Вычисляем матрицу корреляций только для нужных признаков
        corr_matrix = df[features].corr()
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool)) # Маска для верхнего треугольника
        sns.heatmap(corr_matrix,
                    mask=mask,
                    annot=True,
                    fmt='.2f',
                    cmap='coolwarm',
                    center=0,
                    square=True)
        plt.title('Матрица корреляций (верхний треугольник)')
        plt.tight_layout()
        plt.show()

    # Шаг 2: Расчет VIF
    # statsmodels требует добавления константы для расчета VIF
    X = add_constant(df[features])
    vif_data = pd.DataFrame()
    vif_data['Feature'] = X.columns
    
    # Рассчитываем VIF для каждого признака
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    # Убираем константу из результатов (ее VIF будет огромным, но она не нужна)
    vif_data = vif_data[vif_data['Feature'] != 'const']
    vif_data = vif_data.sort_values('VIF', ascending=False).reset_index(drop=True)

    return vif_data


def highlight_high_vif(vif_data: pd.DataFrame, 
                       high_threshold: int = 10, 
                       medium_threshold: int = 5):
    """
    Визуально выделяет признаки с высоким VIF в таблице с градацией цветов.
    
    Parameters:
        vif_data (pd.DataFrame): DataFrame с колонками ['Feature', 'VIF']
        high_threshold (int): Порог для красного выделения (VIF >= 10). По умолчанию 10
        medium_threshold (int): Порог для желтого выделения (5 <= VIF < 10). По умолчанию 5
    
    Returns:
        pd.io.formats.style.Styler: Стилизованный DataFrame для отображения
    """
    
    def _highlight_vif_gradient(s):
        """Внутренняя функция для применения градиентного стиля"""
        styles = []
        for vif_value in s:
            if vif_value >= high_threshold:
                styles.append('background-color: #ffcccc')
            elif medium_threshold <= vif_value < high_threshold:
                styles.append('background-color: #fff2cc')
            else:
                styles.append('')
        return styles
    
    styled_vif = (vif_data.style
                  .apply(_highlight_vif_gradient, subset=['VIF'])
                  .format({'VIF': '{:.2f}'}))
    
    return styled_vif


def save_cleaned_data(df: pd.DataFrame, config) -> dict:
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
    
    # Создание имени файла с датой
    current_date = datetime.now().strftime('%Y%m%d_%H%M')
    filename_base = f"eda_data_{current_date}"
    
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
        
        # Сохранение информации о файлах в конфиг
        config['processed_files']['latest_cleaned'] = {
            'parquet': str(file_paths['parquet']),
            'csv': str(file_paths['csv']),
            'pkl': str(file_paths['pkl']),
            'timestamp': current_date
        }
               
        return config
        
    except Exception as e:
        logging.error(f"Ошибка при сохранении данных: {e}")
        raise