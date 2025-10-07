import os
import sys
import subprocess
import zipfile

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

if project_root not in sys.path:
    sys.path.append(project_root)


def download_and_extract_competition(config: dict,
                                     force_redownload: bool = False):
    """
    Скачивает данные соревнования kaggle и распаковывает zip-архив.
    
    Args:
        competition_name: Название конкурса Kaggle
        download_path: Путь для сохранения данных
        force_redownload: Принудительно перескачать, даже если файлы существуют
    """

    competition_name = config["competition"]["name"]

    # os.path.dirname(__file__) → папка, где лежит текущий скрипт.
    download_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/raw"))

    print(f"[⧗] Загружаю данные соревнования {competition_name} в {download_path}...")
    os.makedirs(download_path, exist_ok=True)

    # Проверяем, есть ли уже распакованные файлы
    if not force_redownload and _has_extracted_files(download_path):
        print("[i] Файлы уже существуют. Используйте force_redownload=True для перезагрузки.")
        return

    try:
        subprocess.check_call(["kaggle", "competitions", "download", "-c",
                              competition_name, "-p", download_path])
    except subprocess.CalledProcessError as e:
        print(f"[✗] Ошибка при загрузке данных: {e}")
        return
    except FileNotFoundError:
        print("[✗] Команда 'kaggle' не найдена.")
        return

    # Распаковываем только новые ZIP-файлы
    for file in os.listdir(download_path):
        if file.endswith(".zip"):
            zip_path = os.path.join(download_path, file)
            _extract_zip_if_needed(zip_path, download_path, force_redownload)
    
    print("[✓] Все готово! Данные скачаны и распакованы.")

    
def _has_extracted_files(directory: str) -> bool:
    """Проверяет, есть ли в папке распакованные файлы (не ZIP)"""
    for file in os.listdir(directory):
        if not file.endswith(".zip"):
            return True
    return False


def _extract_zip_if_needed(zip_path: str, extract_path: str, force: bool = False):
    """Распаковывает ZIP, если файлы ещё не существуют или force=True"""
    zip_name = os.path.basename(zip_path)
    
    # Проверяем, нужно ли распаковывать
    if not force and _zip_already_extracted(zip_path, extract_path):
        print(f"[i] Файлы из {zip_name} уже распакованы")
        return
    
    print(f"[⧗] Распаковываю {zip_name}...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"[✓] Распаковано: {zip_name}")
        
        os.remove(zip_path)
        print(f"[i] Удален архив: {zip_name}")
        
    except zipfile.BadZipFile:
        print(f"[✗] Ошибка: файл {zip_name} не является корректным zip-архивом")
    except Exception as e:
        print(f"[✗] Ошибка при распаковке {zip_name}: {e}")


def _zip_already_extracted(zip_path: str, extract_path: str) -> bool:
    """Проверяет, распакованы ли уже файлы из этого ZIP"""
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            for file_info in zip_ref.infolist():
                extracted_path = os.path.join(extract_path, file_info.filename)
                if not os.path.exists(extracted_path):
                    return False
            return True
    except:
        return False

