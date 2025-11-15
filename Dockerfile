# ===== Imagen base =====
FROM python:3.13-slim

# ===== Config básica =====
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /opt/app-root

# ===== Instalar sistema y uv =====
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential gcc curl \
 && pip install --no-cache-dir uv \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

# ===== Copiar manifests primero =====
COPY pyproject.toml uv.lock* /opt/app-root/
RUN --mount=type=cache,target=/opt/app-root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# ===== Copiar código y terminar instalación =====
COPY . /opt/app-root
RUN --mount=type=cache,target=/opt/app-root/.cache/uv \
    uv sync --frozen --no-dev

# ===== Permisos =====
RUN chgrp -R root /opt/app-root && chmod -R g+rwx /opt/app-root/

COPY entrypoint.sh /opt/app-root/entrypoint.sh
RUN chmod +x /opt/app-root/entrypoint.sh

# ===== Puerto =====
ENV PORT=8030
EXPOSE 8030

# ===== Arranque =====
CMD ["./entrypoint.sh"]