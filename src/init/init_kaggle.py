import os
import subprocess
import sys
import shutil

def ensure_package_installed(package: str):
    """
    Проверяет наличие и устанавливает Python-пакет через pip, если он не установлен.
    """
    try:
        __import__(package.split("==")[0].replace("-", "_"))
        print(f"[✓] {package} уже установлен")
    except ImportError:
        print(f"[⧗] Устанавливаю {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def setup_kaggle_json():
    """
    Проверяет наличие файла kaggle.json и копирует его в ~/.kaggle при необходимости.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    local_kaggle_path = os.path.join(project_root, "kaggle.json")
    user_kaggle_dir = os.path.expanduser("~/.kaggle")
    user_kaggle_path = os.path.join(user_kaggle_dir, "kaggle.json")

    if not os.path.exists(user_kaggle_path):
        if os.path.exists(local_kaggle_path):
            print("[⧗] Копирую kaggle.json в ~/.kaggle/ ...")
            os.makedirs(user_kaggle_dir, exist_ok=True)
            shutil.copy(local_kaggle_path, user_kaggle_path)
            os.chmod(user_kaggle_path, 0o600)
            print("[✓] kaggle.json скопирован и установлен")
        else:
            raise FileNotFoundError("Файл kaggle.json не найден ни в ~/.kaggle/, ни в корне проекта.")
    else:
        print("[✓] kaggle.json уже присутствует в ~/.kaggle/")


def main():
    # Настройка
    competition_name = "skillbox-ml-junior-regression-9"
    download_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/raw"))

    # Проверки и установка зависимостей
    ensure_package_installed("kaggle")
    setup_kaggle_json()

if __name__ == "__main__":
    main()
