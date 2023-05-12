# Utilizamos una imagen de Python 3.9 como base
FROM python:3.11


# Instalar dependencias de sistema
RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get clean

# Establecemos el directorio de trabajo en /app
WORKDIR /app

# Copiamos el archivo requirements.txt al directorio de trabajo
COPY requirements.txt .

# Instalamos las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Instalar Alembic
RUN pip install alembic

# Copiamos el resto de los archivos del proyecto al directorio de trabajo
COPY . .

# Install MySQL client
RUN apt-get update && \
    apt-get install -y default-libmysqlclient-dev && \
    rm -rf /var/lib/apt/lists/*


# Exponemos el puerto 80
EXPOSE 80

# Iniciamos el servidor web
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
