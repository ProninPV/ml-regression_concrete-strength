import numpy as np
import scipy.stats as stats
import pandas as pd
from typing import Optional, Tuple, Dict


def add_concrete_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет в DataFrame два признака:
    - W/C: отношение воды к цементу (Water / Cement)
    - S/C_pct: отношение суперпластификатора к цементу (Superplasticizer / Cement)
    
    Значения, при которых Cement == 0, обрабатываются как NaN.
    """
    df = df.copy()
    
    df['W/C'] = np.where(df['Cement'] != 0, df['Water'] / df['Cement'], np.nan)
    df['Sp/C_pct'] = np.where(df['Cement'] != 0, df['Superplasticizer'] / df['Cement'], np.nan)
    
    return df


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


def preliminary_data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Выполняет предварительную очистку данных в DataFrame.
    
    Эта функция удаляет неинформативные столбцы и дублирующиеся строки
    для подготовки набора данных к дальнейшему анализу и моделированию.
    
    Выполняемые шаги:
    - Удаление столбца 'Id', так как он обычно не несет полезной информации
    - Удаление дублирующихся строк из набора данных
    
    Параметры:
    -----------
    df : pd.DataFrame
        Входной DataFrame, содержащий исходные данные
        
    Возвращает:
    --------
    pd.DataFrame
        Очищенный DataFrame с удаленным столбцом 'Id' и дубликатами строк
    """
    # Удаляем неинформативный столбец ID
    df = df.drop(columns=["Id"])

    # Удаляем дубликаты строк
    df = df.drop_duplicates()

    return df


