#!/bin/bash
set -e

# Cargar variables desde .env si existe
if [ -f "/opt/app-root/.env" ]; then
    echo "🔧 Cargando variables de entorno desde .env"
    export $(grep -v '^#' /opt/app-root/.env | xargs)
else
    echo "⚠️ No se encontró archivo .env, continuando sin variables"
fi

echo "📦 Ejecutando DVC Pull..."
make dvc-pull

echo "🚀 Iniciando API..."
exec uv run -m src.main --host 0.0.0.0 --port 8030