import os
import sys
import subprocess
import zipfile

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

if project_root not in sys.path:
    sys.path.append(project_root)

from src.data import loader

# def download_and_extract_competition(competition_name: str, download_path: str):
#     """
#     Скачивает данные соревнования kaggle и распаковывает zip-архив в указанную папку.
#     """
#     print(f"[⧗] Загружаю данные соревнования {competition_name} в {download_path} ...")

#     # создаёт папку download_path, если её нет
#     # означает «не ругаться, если папка уже существует»
#     os.makedirs(download_path, exist_ok=True)


#     subprocess.check_call(["kaggle", "competitions", "download", "-c",
#                             competition_name, "-p", download_path])

#     # Распаковываем все zip-файлы в папке
#     for file in os.listdir(download_path):
#         if file.endswith(".zip"):
#             zip_path = os.path.join(download_path, file)
#             print(f"[⧗] Распаковываю {file} ...")
#             with zipfile.ZipFile(zip_path, "r") as zip_ref:
#                 zip_ref.extractall(download_path)
#             print(f"[✓] Распаковано: {file}")
#         else:
#             print(f"[i] Пропущен файл: {file}")


def download_and_extract_competition(competition_name: str, download_path: str):
    """
    Скачивает данные соревнования kaggle и распаковывает zip-архив в указанную папку.
    """
    print(f"[⧗] Загружаю данные соревнования {competition_name} в {download_path}...")

    # Создаём папку download_path, если её нет
    # означает «не ругаться, если папка уже существует»
    os.makedirs(download_path, exist_ok=True)

    try:
        subprocess.check_call(["kaggle", "competitions", "download", "-c",
                              competition_name, "-p", download_path])
    except subprocess.CalledProcessError as e:
        print(f"[✗] Ошибка при загрузке данных: {e}")
        return
    except FileNotFoundError:
        print("[✗] Команда 'kaggle' не найдена. Убедитесь, что установлен kaggle API")
        return

    # Распаковываем все zip-файлы в папке
    for file in os.listdir(download_path):
        if file.endswith(".zip"):
            zip_path = os.path.join(download_path, file)
            print(f"[⧗] Распаковываю {file}...")
            try:
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(download_path)
                print(f"[✓] Распаковано: {file}")
                
                # Удаляем zip-файл после распаковки (опционально)
                os.remove(zip_path)
                print(f"[i] Удален архив: {file}")
                
            except zipfile.BadZipFile:
                print(f"[✗] Ошибка: файл {file} не является корректным zip-архивом")
            except Exception as e:
                print(f"[✗] Ошибка при распаковке {file}: {e}")
        else:
            print(f"[i] Пропущен файл: {file}")

def main():
    # Читаем competition_name из config.yaml
    config = loader.load_config()
    competition_name = config["competition"]["name"]

    # os.path.dirname(__file__) → папка, где лежит текущий скрипт.
    download_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/raw"))
    
    # Загрузка и распаковка
    download_and_extract_competition(competition_name, download_dir)

    print("[✓] Все готово! Данные скачаны и распакованы.")

if __name__ == "__main__":
    main()