# Проект: Бинарная классификация для предиктивного обслуживания оборудования

## Описание проекта
Цель проекта — разработать модель машинного обучения, которая
предсказывает, произойдет ли отказ оборудования (Target = 1) или нет
(Target = 0). Результаты работы оформлены в виде Streamlit-приложени
## Датасет
Используется датасет **"AI4I 2020 Predictive Maintenance Dataset"**,
содержащий 10 000 записей с 14 признаками. Подробное описание датасе
можно найти в [документации]
(https://archive.ics.uci.edu/dataset/601/predictive+maintenance+data)
## Установка и запуск
1. Клонируйте репозиторий:

 git clone <https://github.com/Aranfyro/predictive_maintenance_project.git>

2. Установите зависимости:
 pip install -r requirements.txt
3. Запустите приложение:
 streamlit run app.py
## Структура репозитория
- `app.py`: Основной файл приложения.
- `analysis_and_model.py`: Страница с анализом данных и моделью.
- `presentation.py`: Страница с презентацией проекта.
- `requirements.txt`: Файл с зависимостями.
- `data/predictive_maintenance.csv`: Папка с данными.
- `video/video.mp4`: Папка с видео-демонстрацией.
- `README.md`: Описание проекта.
## Видео-демонстрация
[Ссылка на видео](video/video.mp4) или встроенное видео ниже:
<video src="video/video.mp4" controls width="100%"></video>
https://github.com/user-attachments/assets/e27b4f8b-fcdb-41b9-b800-853dbb5de39a

