# Makefile para versionar y subir nuevos archivos a DVC

# Variable para recibir el archivo a versionar, ejemplo:
# make dvc-add FILE=data/processed/clientes_20251001.csv
FILE ?=

dvc-add:
	@if [ -z "$(FILE)" ]; then \
		echo "❌ ERROR: Debes especificar el archivo a versionar. Ejemplo:"; \
		echo "   make dvc-add FILE=data/processed/clientes_20251001.csv"; \
		exit 1; \
	fi
	@echo "🚀 Versionando archivo con DVC: $(FILE)"
	uv run dvc add $(FILE)
	git add $(FILE).dvc .gitignore
	git commit -m "Versionado automático de $(FILE)"
	@echo "⬆️ Subiendo dataset al remoto con DVC..."
	uv run dvc push
	@echo "✅ Archivo versionado y sincronizado con remoto: $(FILE)"

