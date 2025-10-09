import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict
import scipy.stats as stats
from ydata_profiling import ProfileReport
from typing import Optional, Tuple, Dict, Any
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


def detect_outliers(df: pd.DataFrame,
                    config: Optional[Dict],
                    compare_to_standard: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Вычисляет выбросы методом IQR и (опционально) сравнивает значения с эталонными диапазонами.

    Возвращает:
    - summary_df: таблицу характеристик и выбросов
    - outlier_masks_df: DataFrame с булевыми масками выбросов по IQR
    """
    config = config or {}
    standard_dict = config.get("standard_value", {})

    summary = []
    outlier_masks = {}

    numeric_cols = df.select_dtypes(include='number').columns

    for col in numeric_cols:
        series = df[col].dropna()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask_iqr = (df[col] < lower) | (df[col] > upper)
        n_outliers_iqr = mask_iqr.sum()

        col_key = col.replace(' ', '_').replace('/', '_').lower()
        std_min, std_max, n_std_outliers = None, None, None

        if compare_to_standard and col_key in standard_dict:
            std_min = standard_dict[col_key].get("min")
            std_max = standard_dict[col_key].get("max")
            mask_std = (df[col] < std_min) | (df[col] > std_max)
            n_std_outliers = mask_std.sum()            
        else:
            mask_std = None

        summary.append({
            'feature': col,
            'Q1': round(q1, 3),
            'Q3': round(q3, 3),
            'IQR': round(iqr, 3),
            'Lower_Bound': round(lower, 3),
            'Upper_Bound': round(upper, 3),
            'n_outliers_IQR': n_outliers_iqr,
            'standard_min': std_min,
            'standard_max': std_max,
            'n_outliers_standard': n_std_outliers
        })

        outlier_masks[col] = mask_iqr

    summary_df = pd.DataFrame(summary).sort_values('n_outliers_IQR', ascending=False).reset_index(drop=True)
    outlier_masks_df = pd.DataFrame(outlier_masks, index=df.index)

    return summary_df, outlier_masks_df


def analyze_zeros(df: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, Dict]:
    """
    Анализирует признаки с большим количеством нулевых значений и определяет,
    нужно ли добавлять бинарные признаки факта наличия ненулевого значения.
    
    Для каждого признака проводится статистическая проверка (t-test) различия 
    средних значений целевой переменной между группами (значение = 0 и значение > 0).
    Результаты сохраняются в конфигурационный файл.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Датафрейм с данными для анализа
    config : Dict
        Конфигурационный словарь для сохранения результатов
        
    Returns:
    --------
    pd.DataFrame
        Исходный датафрейм с добавленными бинарными признаками (если требуется)
    Dict
        Обновленный конфигурационный словарь
    """
    
    features_to_analyze = ['Blast Furnace Slag', 'Fly Ash', 'Superplasticizer']
    features_to_create = []
    
    print("Анализ значимости нулевых значений для признаков:")
    print("=" * 60)
    
    for feature in features_to_analyze:
        group_0 = df[df[feature] == 0.0]['Strength']
        group_1 = df[df[feature] > 0.0]['Strength']
        
        # 1. Проверяем равенство дисперсий
        _, p_value_levene = stats.levene(group_0, group_1)
        
        # 2. Выбираем и проводим правильный t-test
        if p_value_levene < 0.05:
            _, p_value_ttest = stats.ttest_ind(group_0, group_1, equal_var=False)
            test_type = "Welch's t-test (дисперсии не равны)"
        else:
            _, p_value_ttest = stats.ttest_ind(group_0, group_1, equal_var=True)
            test_type = "Student's t-test (дисперсии равны)"
        
        # 3. Принимаем решение о создании признака
        if p_value_ttest < 0.05:
            print(f"✓ {feature}: p-value = {p_value_ttest:.2e} - Добавляем бинарный признак")
            print(f"  {test_type}, p-value Левене = {p_value_levene:.3f}")
            features_to_create.append(feature)
        else:
            print(f"✗ {feature}: p-value = {p_value_ttest:.3f} - Признак не добавляем")
            print(f"  {test_type}, p-value Левене = {p_value_levene:.3f}")
        
        print("-" * 40)
    
    # Сохраняем результаты в конфиг
    config['binary_features'] = config.get('binary_features', {})
    config['binary_features']['from_zeros'] = features_to_create   

    return config


def validate_eda_config(config: dict, base_path: str = "../..") -> Path:
    """
    Валидирует конфигурацию для EDA-отчёта.
    
    Параметры:
    ----------
    config : dict
        Словарь с конфигурацией
    base_path : str
        Базовый путь для построения директорий
        
    Возвращает:
    -----------
    Path
        Путь для отчётов
        
    Исключения:
    -----------
    ValueError
        Если отсутствуют необходимые ключи в конфигурации
    """
    required_keys = [
        'output/eda_report_dir'
    ]
    
    for key in required_keys:
        keys = key.split('/')
        current = config
        for k in keys:
            if k not in current:
                raise ValueError(f"Отсутствует обязательный ключ в конфигурации: {key}")
            current = current[k]
    
    report_dir = Path(base_path) / config["output"]["eda_report_dir"]
    
    return report_dir


def should_generate_report(output_file: Path, resave: bool) -> bool:
    """
    Проверяет, нужно ли генерировать отчёт.
    
    Параметры:
    ----------
    output_file : Path
        Путь к файлу отчёта
    resave : bool
        Флаг перезаписи существующего отчёта
        
    Возвращает:
    -----------
    bool
        True если нужно генерировать отчёт, False если можно пропустить
    """
    if output_file.exists() and not resave:
        return False
    return True


def analyze_missing_values(data: pd.DataFrame) -> Dict[str, int]:
    """
    Анализирует пропущенные значения в данных.
    
    Параметры:
    ----------
    data : pd.DataFrame
        Данные для анализа
        
    Возвращает:
    -----------
    Dict[str, int]
        Словарь с количеством пропусков по колонкам
    """
    missing = data.isnull().sum()
    missing_total = missing.sum()
    missing_columns = missing[missing > 0]
    
    missing_info = {
        'total_missing': int(missing_total),
        'columns_with_missing': missing_columns.to_dict(),
        'num_columns_with_missing': len(missing_columns)
    }
    
    if missing_total > 0:
        print(f"Обнаружены пропуски в {missing_info['num_columns_with_missing']} колонках:")
        for col, count in missing_columns.items():
            print(f" - {col}: {count} пропусков")
    else:
        print("Пропусков в данных не обнаружено.")
    
    return missing_info


def extract_profile_warnings(profile: ProfileReport) -> list:
    """
    Извлекает и анализирует предупреждения из профиля отчёта.
    
    Параметры:
    ----------
    profile : ProfileReport
        Сгенерированный профиль отчёта
        
    Возвращает:
    -----------
    list
        Список предупреждений
    """
    try:
        description = profile.get_description()
        warnings = getattr(description, "warnings", [])
        
        if warnings:
            print(f"Обнаружено {len(warnings)} предупреждений из отчета:")
            for warning in warnings:
                print(f" - {warning}")
        else:
            print("Предупреждения ydata-profiling отсутствуют.")
            
        return warnings
    except Exception as e:
        print(f"Ошибка при извлечении предупреждений: {str(e)}")
        return []


def generate_profile_report(data: pd.DataFrame, dataset_name: str, output_file: Path) -> ProfileReport:
    """
    Генерирует EDA-отчёт и сохраняет его в файл.
    
    Параметры:
    ----------
    data : pd.DataFrame
        Данные для анализа
    dataset_name : str
        Название набора данных
    output_file : Path
        Путь для сохранения отчёта
        
    Возвращает:
    -----------
    ProfileReport
        Сгенерированный профиль отчёта
    """
    print(f"Генерация EDA отчета для '{dataset_name}'...")
    
    profile = ProfileReport(data, title=f"{dataset_name} EDA Report", explorative=True)
    profile.to_file(output_file)
    
    print(f"Отчёт сохранён: {output_file}")
    return profile


def eda_report(data: pd.DataFrame, dataset_name: str, config: dict, resave: bool = True, base_path: str = "../..") -> None:
    """
    Генерирует EDA-отчёт с использованием ydata-profiling и сохраняет его в HTML-файл.
    
    Параметры:
    ----------
    data : pd.DataFrame
        Таблица с данными для анализа.
    dataset_name : str
        Название набора данных.
    config : dict
        Словарь с конфигурацией.
    resave : bool
        Флаг перезаписи существующего отчёта.
    base_path : str
        Базовый путь для построения директорий
        
    Возвращает:
    -----------
    None
    """
    try:
        # Валидация конфигурации и получение путей
        report_dir = validate_eda_config(config, base_path)
        
        # Создание директории для отчётов
        report_dir.mkdir(parents=True, exist_ok=True)
        output_file = report_dir / f"{dataset_name}_eda_report.html"
        
        # Проверка необходимости генерации отчёта
        if not should_generate_report(output_file, resave):
            print(f"Отчёт уже существует: {output_file}. Пропускаем генерацию.")
            return
        
        # Анализ пропущенных значений
        missing_info = analyze_missing_values(data)
        
        # Генерация отчёта
        profile = generate_profile_report(data, dataset_name, output_file)
        
        # Анализ предупреждений из отчёта
        warnings = extract_profile_warnings(profile)
        
        print("EDA-отчёт успешно сгенерирован и проанализирован")
        
    except Exception as e:
        print(f"Ошибка при генерации EDA-отчета: {str(e)}")
        raise


def calculate_trend_metrics(df: pd.DataFrame, 
                          features: list,
                          target: str,
                          config: Dict[str, Any]) -> pd.DataFrame:
    """
    Вычисляет метрики трендов для признаков.
    """
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
    
    trend_funcs = [linear_func, log_func, sqrt_func, reciprocal_func, square_func]
    trend_names = config['trend_settings']['names']
    
    results_data = []
    
    for feature in features:
        x_data = df[feature].values
        y_data = df[target].values
        
        sort_idx = np.argsort(x_data)
        x_sorted = x_data[sort_idx]
        y_sorted = y_data[sort_idx]
        
        r2_scores = {}
        
        for func, name in zip(trend_funcs, trend_names):
            try:
                popt, _ = curve_fit(func, x_sorted, y_sorted, maxfev=5000)
                y_trend = func(x_sorted, *popt)
                r2 = r2_score(y_sorted, y_trend)
                r2_scores[name] = r2
            except Exception:
                r2_scores[name] = -np.inf
        
        results_data.append({
            'feature': feature,
            'linear_r2_score': r2_scores.get('Linear', np.nan),
            'log_r2_score': r2_scores.get('Log', np.nan),
            'sqrt_r2_score': r2_scores.get('Sqrt', np.nan),
            'reciprocal_r2_score': r2_scores.get('1/x', np.nan),
            'square_r2_score': r2_scores.get('x²', np.nan)
        })
    
    return pd.DataFrame(results_data)


def select_best_transformations(metrics_df: pd.DataFrame, alpha: float = 0.2) -> pd.DataFrame:
    """
    Выбирает лучшие преобразования на основе метрик.
    """
    trend_columns = ['linear_r2_score', 'log_r2_score', 'sqrt_r2_score', 
                    'reciprocal_r2_score', 'square_r2_score']
    trend_names = ['Linear', 'Log', 'Sqrt', '1/x', 'x²']
    
    results = []
    
    for _, row in metrics_df.iterrows():
        r2_scores = {name: row[col] for name, col in zip(trend_names, trend_columns)}
        
        best_trend_name = max(r2_scores.items(), key=lambda x: x[1])[0]
        best_r2 = r2_scores[best_trend_name]
        linear_r2 = r2_scores.get('Linear', -np.inf)
        
        if best_trend_name != 'Linear' and linear_r2 >= best_r2 * (1 - alpha):
            final_best_trend = 'Linear'
            final_best_r2 = linear_r2
        else:
            final_best_trend = best_trend_name
            final_best_r2 = best_r2
        
        result_row = row.to_dict()
        result_row.update({
            'best_transformation': final_best_trend,
            'best_r2_score': final_best_r2
        })
        results.append(result_row)
    
    return pd.DataFrame(results)