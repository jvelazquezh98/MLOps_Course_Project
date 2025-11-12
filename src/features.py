# src/features.py
from pathlib import Path
from loguru import logger
from tqdm import tqdm
import typer
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Importar configuraciones - FORMA CORREGIDA
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "online_news_modified.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv"
):
    """
    Genera features para el dataset de noticias online
    """
    logger.info("🎯 Iniciando generación de features...")
    
    try:
        # 1. CARGAR DATOS
        logger.info(f"📂 Cargando datos desde: {input_path}")
        df = pd.read_csv(input_path)
        logger.info(f"✅ Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
        logger.info(f"📊 Columnas disponibles: {len(df.columns)}")
        logger.info(f"🔧 Tipo de datos actual: {df.dtypes.unique()}")
        
        # 2. LIMPIAR COLUMNA mixed_type_col
        logger.info("🔄 Limpiando columna mixed_type_col...")
        df['mixed_type_col_clean'] = df['mixed_type_col'].replace({
            'bad': 0,
            'unknown': 1
        })
        df['mixed_type_col_clean'] = pd.to_numeric(df['mixed_type_col_clean'], errors='coerce')
        logger.info(f"✅ mixed_type_col convertida: {df['mixed_type_col_clean'].notnull().sum()} valores válidos")
        
        # 3. ELIMINAR COLUMNA ORIGINAL PROBLEMÁTICA
        df = df.drop('mixed_type_col', axis=1)
        logger.info("🗑️ Columna mixed_type_col original eliminada")
        
        # 4. CONVERTIR TODAS LAS COLUMNAS A NUMÉRICAS
        logger.info("🔄 Convirtiendo todas las columnas a numéricas...")
        numeric_columns = [col for col in df.columns if col != 'url']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        logger.info(f"✅ Todas las columnas convertidas a numéricas")
        logger.info(f"📊 Tipos de datos después de conversión: {df.dtypes.unique()}")
        
        # 5. MANEJAR VALORES NULOS
        logger.info("🔄 Manejando valores nulos...")
        nulls_before = df.isnull().sum().sum()
        logger.info(f"Valores nulos totales antes: {nulls_before}")
        
        # Eliminar filas con muchos valores nulos
        df = df.dropna(thresh=df.shape[1]//2)
        logger.info(f"📊 Dataset después de eliminar filas muy nulas: {df.shape[0]} filas")
        
        # Llenar valores nulos con mediana
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
        
        nulls_after = df.isnull().sum().sum()
        logger.info(f"✅ Valores nulos después: {nulls_after}")
        logger.info(f"🗑️ Valores nulos eliminados/rellenados: {nulls_before - nulls_after}")
        
        # 6. GUARDAR EL DATASET PROCESADO
        logger.info(f"💾 Guardando dataset procesado en: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.success(f"✅ Dataset guardado exitosamente en: {output_path}")
        
        # Resumen final
        logger.success("🎉 Proceso de feature engineering completado!")
        logger.info(f"📈 Resumen final:")
        logger.info(f"   - Filas: {df.shape[0]}")
        logger.info(f"   - Columnas: {df.shape[1]}")
        logger.info(f"   - Valores nulos procesados: {nulls_before - nulls_after}")
        logger.info(f"   - Archivo guardado: {output_path}")
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Error generando features: {e}")
        raise

if __name__ == "__main__":
    app()