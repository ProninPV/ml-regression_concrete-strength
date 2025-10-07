import os
import pandas as pd
from typing import List, Optional
from datetime import datetime


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