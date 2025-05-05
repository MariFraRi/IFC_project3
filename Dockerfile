# Utiliza una imagen oficial de Python
FROM python:3.10-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia los archivos necesarios
COPY . .

# Instala las dependencias
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Expone el puerto por defecto de Dash
EXPOSE 8050

# Comando para ejecutar la aplicación Dash
CMD ["python", "app.py"]

