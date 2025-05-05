# 📊 Dashboard de Inversiones IFC – Entregable #3
**Autor:** Mariana Franco & Samuel Bermudez
**Fecha:** Mayo 2025
**Link de render:** https://ifc-project3.onrender.com


Este proyecto consiste en el desarrollo de un Dashboard interactivo utilizando **Dash** (Python) y una base de datos **PostgreSQL** para el análisis y visualización de datos de inversiones del IFC. El objetivo es realizar un análisis exploratorio y visualización de resultados de forma dinámica, con un despliegue empaquetado en **Docker**.

---

## 🧰 Tecnologías Utilizadas

- **Python 3**
- **Dash** y **Plotly** para visualización
- **PostgreSQL** para almacenamiento de datos
- **SQLAlchemy** para conexión con la base de datos
- **Docker** y `docker-compose` para el despliegue
- **Pandas**, **Scikit-learn**, entre otros

---

## 📁 Estructura del Proyecto

```
├── app.py                    # Código principal del dashboard (Dash)
├── cargar_postgres.py        # Script para cargar los datos CSV a PostgreSQL
├── consultas_postgres.py     # Funciones de consulta a PostgreSQL
├── datos.csv                 # Dataset original a cargar
├── Dockerfile                # Imagen personalizada para la app Dash
├── docker-compose.yml        # Orquestación de servicios con PostgreSQL
├── requirements.txt          # Librerías necesarias
├── .dockerignore             # Archivos ignorados por Docker
└── capturas/                 # Evidencias visuales del funcionamiento
```

---

## 🗃️ Descripción de Componentes

- `cargar_postgres.py`: lee el archivo `datos.csv` y lo carga en la tabla `datosifc_csv` de PostgreSQL.
- `consultas_postgres.py`: funciones para consultar datos desde la base y usarlas en el dashboard.
- `app.py`: construcción del dashboard con filtros, visualizaciones y tabla de datos.
- `docker-compose.yml`: define dos servicios (PostgreSQL y la aplicación Dash).
- `Dockerfile`: contiene instrucciones para construir la imagen de la app.
- `requirements.txt`: dependencias del entorno de Python.

---

## 📊 Funcionalidades del Dashboard

### 🟦 Pestaña: Resumen de Inversiones

- Filtros por **Industria** y **Estado del Proyecto**
- Visualizaciones:
  - Boxplot de inversión por industria
  - Barras por categoría ambiental
  - Conteo de proyectos por industria
  - Histograma de inversiones
  - Serie de tiempo por año
  - Inversión total por estado

### 🌍 Pestaña: Análisis por País

- Gráfica interactiva con el **Top 10 países** por inversión (orden ascendente o descendente)

### 📋 Pestaña: Tabla de Datos

- Tabla con filtros nativos y ordenamiento sobre los registros cargados desde la base de datos

---


### 3. Cargar los datos

Desde el contenedor o tu terminal local, ejecutar:

```bash
python cargar_postgres.py
```
Hay un ligero cambio cuando se quiere desplegar con docker local y en railway en la manera en como lee los datos, pues render usa su propio environment de postgres.
---

## 📷 Capturas

Se incluye el directorio `/capturas` con imágenes del funcionamiento del dashboard.

---

## 🔍 Consideraciones Finales

- El análisis se centra en un **EDA (Análisis Exploratorio de Datos)**, visualizaciones descriptivas y tendencias.
- El modelo puede extenderse para incluir predicciones o métricas de impacto.
- Se respetó el template oficial recomendado por el curso:
  [https://dataviz-template-dash-12.onrender.com](https://dataviz-template-dash-12.onrender.com)

---

## 📎 Enlaces de Referencia

- Repositorio base: [DATAVIZ Template Dash](https://github.com/Kalbam/DATAVIZ_Template_Dash)
- Galería de ejemplos: [Plotly Dash Examples](https://plotly.com/examples/)
- Documentación útil:
  - [Statsmodels](https://www.statsmodels.org/stable/examples/index.html#linear-regression-models)
  - [Scikit-Learn](https://www.datacamp.com/es/blog/category/machine-learning)

---
