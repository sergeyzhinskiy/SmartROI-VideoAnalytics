# 👁️ SmartROI-VideoAnalytics

Система распределенной интеллектуальной видеоаналитики в реальном времени. Проект разработан для детекции лиц на больших дистанциях (**4–5 метров**) на оборудовании без дискретных видеокарт (**CPU-only**).

Реализована клиент-серверная архитектура: тяжелые вычисления (AI/CV) происходят на сервере, а динамическое управление, интерактивный интерфейс, воспроизведение уведомлений и сохранение логов осуществляются на стороне клиента в браузере.

##  Ключевые возможности

*   **Детекция лиц на расстоянии 4–5 метров:** Использование оптимизированной легковесной модели `YOLOv8-Face` на базе архитектуры Ultralytics.
*   **Динамическая зона контроля (ROI):** Пользователь может мышкой рисовать и менять зону отслеживания прямо на HTML-странице в реальном времени без перезапуска сервера.
*   **Анализ возраста и эмоций:** Распознавание характеристик лица с помощью предобученных нейросетей `DeepFace` (VGG-Face / Facenet).
*   **Высокая оптимизация (Высокий FPS на CPU):** 
    *   Нейросеть YOLO обрабатывает только выделенную область (ROI), а не весь HD-кадр.
    *   Тяжелый анализ атрибутов лица (DeepFace) запускается асинхронно раз в 5 кадров.
*   **Автоматические триггеры на клиенте:** При фиксации входа/выхода человека браузер клиента автоматически скачивает скриншот с метаданными (таймкод в названии) и воспроизводит соответствующий аудиофайл (`1.mp3` / `2.mp3`).
*   **Кроссплатформенность:** Серверная часть протестирована и стабильно работает на Windows и Linux (включая безмониторные Ubuntu Server по SSH).

---

##  Архитектура проекта

```text
📁 cv/
├── server.py               # Flask-сервер, захват видеопотока, YOLOv8 + DeepFace
├── yolov8n-face.pt         # Веса модели детектора лиц (6 МБ)
├── 📁 static/
│   ├── 1.mp3               # Аудио: Триггер появления человека в зоне
│   └── 2.mp3               # Аудио: Триггер исчезновения человека из зоны
└── 📁 templates/
    └── index.html          # Клиентский интерфейс (JS, Canvas ROI, Логика триггеров)
```

---

##  Системные требования и зависимости

###  Для Linux систем (Ubuntu/Debian)
Перед установкой Python-пакетов необходимо установить системные графические библиотеки, так как на серверных дистрибутивах Linux отсутствует встроенный графический движок:
```bash
sudo apt update && sudo apt install -y libgl1-mesa-glx libglib2.0-0
```

###  Python библиотеки
Установите зависимости с помощью пакетного менеджера `pip`:
```bash
pip install flask ultralytics opencv-python deepface
```

---

##  Предварительная загрузка весов (Оффлайн-режим для Хакатонов)

На площадках хакатонов часто возникают проблемы с интернетом или блокировками серверов хранения моделей. Рекомендуется скачать веса заранее:

1.  **Модель лица YOLOv8:** Скачайте файл [yolov8n-face.pt]([https://github.com](https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt)) и положите его в корень проекта.
2.  **Модели DeepFace:** Скачайте файлы моделей [age_model_weights.h5]([https://github.com](https://release-assets.githubusercontent.com/github-production-release-asset/382368840/88f8b280-db87-11eb-9f3a-a25179fc0064?sp=r&sv=2018-11-09&sr=b&spr=https&se=2026-09-09T22%3A56%3A11Z&rscd=attachment%3B+filename%3Dage_model_weights.h5&rsct=application%2Foctet-stream&skoid=96c2d410-5711-43a1-aedd-ab1947aa7ab0&sktid=398a6654-997b-47e9-b12b-9515b896b4de&skt=2026-09-09T21%3A56%3A01Z&ske=2026-09-09T22%3A56%3A11Z&sks=b&skv=2018-11-09&sig=uVTC2bv9KTEkilZ5bxQtZ9JhVzuEPg6%2Bsg19D93LPes%3D&jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmVsZWFzZS1hc3NldHMuZ2l0aHVidXNlcmNvbnRlbnQuY29tIiwia2V5Ijoia2V5MSIsImV4cCI6MTc4ODk5NDY5NiwibmJmIjoxNzg4OTkxMDk2LCJwYXRoIjoicmVsZWFzZWFzc2V0cHJvZHVjdGlvbi5ibG9iLmNvcmUud2luZG93cy5uZXQifQ.TivGIYAVt6dxX9Cvu15k-lsFjVHA7l_9Ec9B5CFTfC0&response-content-disposition=attachment%3B%20filename%3Dage_model_weights.h5&response-content-type=application%2Foctet-stream)) и [facial_expression_model_weights.h5]([https://github.com](https://release-assets.githubusercontent.com/github-production-release-asset/382368840/0c6bd100-db93-11eb-819c-0b61869108d3?sp=r&sv=2018-11-09&sr=b&spr=https&se=2026-09-09T22%3A33%3A55Z&rscd=attachment%3B+filename%3Dfacial_expression_model_weights.h5&rsct=application%2Foctet-stream&skoid=96c2d410-5711-43a1-aedd-ab1947aa7ab0&sktid=398a6654-997b-47e9-b12b-9515b896b4de&skt=2026-09-09T21%3A32%3A56Z&ske=2026-09-09T22%3A33%3A55Z&sks=b&skv=2018-11-09&sig=eJCbzzjXn%2BKDuhG0ZdFPFyg4LiK2Degvggpp1Z3oCxc%3D&jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmVsZWFzZS1hc3NldHMuZ2l0aHVidXNlcmNvbnRlbnQuY29tIiwia2V5Ijoia2V5MSIsImV4cCI6MTc4ODk5MTQxMSwibmJmIjoxNzg4OTkxMTExLCJwYXRoIjoicmVsZWFzZWFzc2V0cHJvZHVjdGlvbi5ibG9iLmNvcmUud2luZG93cy5uZXQifQ.sJQb0_IJK5CtwCNInZLZIIOdbca-uCjD2XDH80eJ_Vc&response-content-disposition=attachment%3B%20filename%3Dfacial_expression_model_weights.h5&response-content-type=application%2Foctet-stream)). Поместите их в системную директорию весов DeepFace:
    *   **Windows:** `%USERPROFILE%\.deepface\weights\`
    *   **Linux:** `~/.deepface/weights/`

---

##  Быстрый запуск

1. Подключите веб-камеру к вашему серверу (ПК/ноутбуку).
2. Запустите скрипт сервера:
   ```bash
   python server.py
   ```
3. Откройте браузер на клиенте (это может быть любой компьютер или планшет в той же локальной сети) по адресу:
   ```text
   http://localhost:5000   # Если запускается локально
   http://<IP_СЕРВЕРА>:5000  # Если запускается на удаленном сервере/Linux
   ```
4. **Важно:** Нажмите большую синюю кнопку на веб-странице — **«АКТИВИРОВАТЬ СИСТЕМУ»**. Это необходимо для разблокировки строгих политик безопасности браузера (браузеры запрещают воспроизводить звуки и скачивать файлы в фоновом режиме без явного согласия пользователя).
5. Зажмите левую кнопку мыши на видео и растяните рамку, чтобы задать **динамическую зону контроля**.

---

##  Технические детали оптимизации

*   **Разрешение:** Поток принудительно инициализируется в HD качестве `1280x720` через `cv2.CAP_PROP_FRAME_WIDTH`. Это критично для высокой точности распознавания мелких объектов (лиц) на расстоянии 5 метров.
*   **Снижение нагрузки на CPU:** Параметр `imgsz=320` в `model.predict` сжимает область детекции перед передачей в YOLO, что дает прирост скорости в 2.5 раза на слабых процессорах без потери дальнобойности.
*   **Устранение оконных конфликтов:** В коде отсутствуют вызовы `cv2.imshow()` и `cv2.waitKey()`, что позволяет запускать бэкенд в Docker или фоновых процессах (systemd/screen) на Linux-серверах без графической оболочки (X11).

##  Лицензия
Проект распространяется под лицензией MIT.
