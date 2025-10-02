# Локальный запуск (Windows 11 + WSL2 и Ubuntu 22.04+)


### Windows 11
- [Docker Desktop]
- Включённый WSL2 и Linux-дистрибутив (Ubuntu рекомендуем)
- Для GPU: в Docker Desktop включить **Use the WSL 2 based engine** и **Use integrated GPU** (если доступно), установлены драйверы NVIDIA и **NVIDIA Container Toolkit** в WSL2-дистрибутиве.

### Ubuntu 22.04+
- Установлен Docker Engine
- Для GPU: установлен **NVIDIA Container Toolkit (NCTK)**  
  

### Запуск GPU:
  ```bash
    docker compose -f docker-compose.yml -f docker-compose.gpu.yml build
    docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
  ```

### Запуск CPU:
  ```bash
    docker compose build
    docker compose up -d
  ```
  
### WEB:
`http://localhost:8000/`