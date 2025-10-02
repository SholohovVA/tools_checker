Детекция инструментов для АФЛТ-Системс

###Локальный запуск:
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000

Web:
http://127.0.0.1:8000/

Swagger:
http://127.0.0.1:8000/docs

###Запуск через докер:

Windows 11
- [Docker Desktop]
- Включённый WSL2 и Linux-дистрибутив (Ubuntu рекомендуем)
- Для GPU: в Docker Desktop включить **Use the WSL 2 based engine** и **Use integrated GPU** (если доступно), установлены драйверы NVIDIA и **NVIDIA Container Toolkit** в WSL2-дистрибутиве.

Ubuntu 22.04+
- Установлен Docker Engine
- Для GPU: установлен **NVIDIA Container Toolkit (NCTK)**


Запуск GPU:
    docker compose -f docker-compose.yml -f docker-compose.gpu.yml build
    docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d

Запуск CPU:
    docker compose build
    docker compose up -d