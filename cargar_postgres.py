import pandas as pd
from sqlalchemy import create_engine
import os

# Cargar datos
file_path = "datos.csv"
df = pd.read_csv(file_path)
print("Primeros registros del archivo:")
print(df.head())

# Leer configuración desde variables de entorno
usuario = os.getenv("DB_USER", "postgres")
contrasena = os.getenv("DB_PASSWORD", "clave123")
host = os.getenv("DB_HOST", "localhost")
puerto = os.getenv("DB_PORT", "5432")
base_datos = os.getenv("DB_NAME", "proyecto_personal")

engine = create_engine(f"postgresql+psycopg2://{usuario}:{contrasena}@{host}:{puerto}/{base_datos}")

df.to_sql("datosifc_csv", engine, if_exists="replace", index=False)
print("\nDatos insertados correctamente en la tabla 'datosifc_csv'.")
