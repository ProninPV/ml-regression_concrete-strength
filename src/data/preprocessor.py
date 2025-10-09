import numpy as np
import pandas as pd


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
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])

    # Удаляем дубликаты строк
    df = df.drop_duplicates()

    return df


def checking_order_of_features(df_train: pd.DataFrame,
                               df_test: pd.DataFrame) -> bool:
    """
    Проверяет совпадение порядка и состава признаков в тренировочном и тестовом наборах данных.
    
    Parameters
    ----------
    df_train : pd.DataFrame
        Тренировочный датафрейм, содержащий целевую переменную "Strength"
    df_test : pd.DataFrame
        Тестовый датафрейм
    
    Returns
    -------
    bool
        True если порядок признаков совпадает, False если отличается
    """
    # Создание массива из признаков и массива из целевой переменной
    X = df_train.drop(columns=["Strength"])

    # Сравниваем порядок признаков в тренировочном и тестовом датасете
    train_features = list(X.columns)
    test_features = list(df_test.columns)
        
    if list(X.columns) == list(df_test.columns):
        print("Порядок признаков совпадает")
        return True
    else:
        print("Порядок признаков отличается")
        print(f"Признаки train: {train_features}")
        print(f"Признаки test: {test_features}")
        return False       