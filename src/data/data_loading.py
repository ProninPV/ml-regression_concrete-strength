import os
import sys
import subprocess
import zipfile

from utils import ml_utils

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

if project_root not in sys.path:
    sys.path.append(project_root)


from src.utils import config_loader, ml_utils


def download_and_extract_competition(competition_name: str, download_path: str):
    """
    Скачивает данные соревнования kaggle и распаковывает zip-архив в указанную папку.
    """
    print(f"[⧗] Загружаю данные соревнования {competition_name} в {download_path} ...")

    # создаёт папку download_path, если её нет
    # означает «не ругаться, если папка уже существует»
    os.makedirs(download_path, exist_ok=True)


    subprocess.check_call(["kaggle", "competitions", "download", "-c",
                            competition_name, "-p", download_path])

    # Распаковываем все zip-файлы в папке
    for file in os.listdir(download_path):
        if file.endswith(".zip"):
            zip_path = os.path.join(download_path, file)
            print(f"[⧗] Распаковываю {file} ...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(download_path)
            print(f"[✓] Распаковано: {file}")
        else:
            print(f"[i] Пропущен файл: {file}")

def main():
    # Читаем competition_name из config.yaml
    config = ml_utils.load_config()
    competition_name = config["competition"]["name"]

    # os.path.dirname(__file__) → папка, где лежит текущий скрипт.
    download_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/raw"))
    
    # Загрузка и распаковка
    download_and_extract_competition(competition_name, download_dir)

    print("[✓] Все готово! Данные скачаны и распакованы.")

if __name__ == "__main__":
    main()