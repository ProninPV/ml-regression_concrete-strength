import subprocess
import sys

# Расширенный список необходимых библиотек
required_packages = [
    "pandas",
    "notebook",
    "jupyterlab",
    "scikit-learn",
    "matplotlib",
    "scipy",
    "seaborn",
    "numpy<=2.1",      # Явная фиксация стабильной версии
    "tqdm",
    "joblib",
    "pyyaml",
    "xgboost",
    "lightgbm",
    "optuna",
    "mlflow",
    "openpyxl",
    "wandb"
    "ydata-profiling"
    "numba"
]

def install_if_missing(package):
    """Проверяет наличие и устанавливает пакет, если он отсутствует."""
    try:
        # импортирует библиотеку, если получилось все ок
        __import__(package.split("==")[0].replace("-", "_"))
        print(f"[✓] {package} уже установлен")
    except ImportError:
        # если импорт не удался, то устанавливает пакет 
        print(f"[⧗] Устанавливаю {package}...")
        # subprocess.check_call([...]) запускает процесс установки:
        # sys.executable — путь к текущему Python-интерпретатору.
        # -m pip install package — команда для установки пакета.
        # -m - Находит модуль pip в своём окружении и запускает его как скрипт
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

if __name__ == "__main__":
    for pkg in required_packages:
        install_if_missing(pkg)

    print()
    print('Скрипт завершен!')