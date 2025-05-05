import pandas as pd
from sqlalchemy import create_engine
import os

# Diccionarios de traducción
traducciones_industria = {
    "Oil, Gas, and Mining": "Petróleo, Gas y Minería",
    "Telecommunications, Media, and Technology": "Telecomunicaciones, Medios y Tecnología",
    "Health and Education": "Salud y Educación",
    "Tourism, Retail, and Property": "Turismo, Comercio y Propiedad",
    "Funds": "Fondos",
    "other": "Otro",
    "Manufacturing": "Manufactura",
    "Agribusiness and Forestry": "Agroindustria y Silvicultura",
    "Infrastructure": "Infraestructura",
    "Financial Institutions": "Instituciones Financieras"
}

traducciones_estado = {
    "Active": "Activo",
    "Completed": "Completado",
    "Pending Signing": "Firma Pendiente",
    "Pending Disbursement": "Desembolso Pendiente",
    "Pending Approval": "Aprobación Pendiente",
    "Hold": "En Espera"
}

def conectar_postgres():
    usuario = os.getenv("DB_USER", "postgres")
    contrasena = os.getenv("DB_PASSWORD", "clave123")
    host = os.getenv("DB_HOST", "localhost")
    puerto = os.getenv("DB_PORT", "5432")
    base_datos = os.getenv("DB_NAME", "proyecto_personal")
    engine = create_engine(f"postgresql+psycopg2://{usuario}:{contrasena}@{host}:{puerto}/{base_datos}")
    return engine

def obtener_datos_base():
    engine = conectar_postgres()
    df = pd.read_sql("SELECT * FROM datosifc_csv", engine)
    df["industria"] = df["industria"].replace(traducciones_industria)
    df["estado"] = df["estado"].replace(traducciones_estado)
    return df

def inversiones_por_industria():
    engine = conectar_postgres()
    df = pd.read_sql("""
        SELECT industria, total_inversion_ifc_aprobada_junta_millones_usd
        FROM datosifc_csv
        WHERE industria IS NOT NULL
    """, engine)
    df["industria"] = df["industria"].replace(traducciones_industria)
    return df

def suma_inversiones_por_categoria():
    engine = conectar_postgres()
    df = pd.read_sql("""
        SELECT categoria_ambiental, SUM(total_inversion_ifc_aprobada_junta_millones_usd) AS total
        FROM datosifc_csv
        GROUP BY categoria_ambiental
    """, engine)
    return df

def conteo_por_industria():
    engine = conectar_postgres()
    df = pd.read_sql("""
        SELECT industria, COUNT(*) as cantidad
        FROM datosifc_csv
        GROUP BY industria
        ORDER BY cantidad DESC
    """, engine)
    df["industria"] = df["industria"].replace(traducciones_industria)
    return df

def inversiones_por_estado():
    engine = conectar_postgres()
    df = pd.read_sql("""
        SELECT estado, SUM(total_inversion_ifc_aprobada_junta_millones_usd) AS total
        FROM datosifc_csv
        GROUP BY estado
        ORDER BY total DESC
    """, engine)
    df["estado"] = df["estado"].replace(traducciones_estado)
    return df

def conteo_por_anio():
    engine = conectar_postgres()
    query = """
        SELECT EXTRACT(YEAR FROM fecha_divulgada::timestamp) AS anio, COUNT(*) AS proyectos
        FROM "datosifc_csv"
        GROUP BY anio
        ORDER BY anio
    """
    return pd.read_sql(query, engine)

def top_paises_inversion(orden='desc'):
    engine = conectar_postgres()
    orden_sql = "DESC" if orden == 'desc' else "ASC"
    df = pd.read_sql(f"""
        SELECT pais, SUM(total_inversion_ifc_aprobada_junta_millones_usd) AS total
        FROM datosifc_csv
        GROUP BY pais
        ORDER BY total {orden_sql}
        LIMIT 10
    """, engine)
    return df
