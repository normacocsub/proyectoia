# Proyecto FastAPI con Docker Compose

## Proyecto creado con MySql

Requisitos previos
 - Docker y Docker Compose deben estar instalados en tu sistema.

# Ejecución con Docker Compose
1. Clona este repositorio en tu máquina local.
2. Crea un archivo .env en la raíz del proyecto con las siguientes variables de entorno:

```MYSQL_DATABASE=nombre_de_la_bd
MYSQL_USER=nombre_de_usuario
MYSQL_PASSWORD=contraseña_del_usuario
MYSQL_ROOT_PASSWORD=contraseña_del_usuario_root
MYSQL_HOST=host de la base de datos
DB_URL=mysql+pymysql://nombre_de_usuario:contraseña_del_usuario@db/nombre_de_la_bd
```
Reemplaza los valores con los que quieras utilizar para tu base de datos y usuario.

3. En el directorio raíz del proyecto, ejecuta el siguiente comando para iniciar los contenedores:

```docker-compose up --build ```

Esto creará los contenedores necesarios y ejecutará la aplicación en http://localhost:80.

4. Si necesitas detener los contenedores, ejecuta:
 ```docker-compose down```


# Ejecución sin Docker

1. Clona este repositorio en tu máquina local.
2. Crea un archivo .env en la raíz del proyecto con las siguientes variables de entorno:

```MYSQL_DATABASE=nombre_de_la_bd
MYSQL_USER=nombre_de_usuario
MYSQL_PASSWORD=contraseña_del_usuario
MYSQL_ROOT_PASSWORD=contraseña_del_usuario_root
MYSQL_HOST=host de la base de datos
DB_URL=mysql+pymysql://nombre_de_usuario:contraseña_del_usuario@db/nombre_de_la_bd
```

Reemplaza los valores con los que quieras utilizar para tu base de datos y usuario.

3. Crea y activa un entorno virtual de Python:
linux / mac
```python3 -m venv myenv
source myenv/bin/activate
```
windows

```python3 -m venv myenv
myenv\Scripts\activate.bat
```

Esto activará el entorno virtual y cambiará el símbolo del sistema para mostrar el nombre del entorno virtual actual.

4. Instala las dependencias utilizando pip y el archivo requirements.txt, ejecutando el siguiente comando:
```pip install -r requirements.txt```

5. En el directorio raíz del proyecto, ejecuta el siguiente comando para iniciar la aplicación:

```uvicorn app.main:app --reload```
Esto iniciará el servidor y ejecutará la aplicación en http://localhost:8000.

6. Si necesitas detener la aplicación, presiona Ctrl + C en la terminal.

7. Cuando hayas terminado de trabajar con el entorno virtual, puedes desactivarlo ejecutando el siguiente comando:
```deactivate```
Esto volverá a establecer el símbolo del sistema para que muestre la ubicación de tu carpeta actual, en lugar del entorno virtual.

# Test TestClient

Los test son de las rutas y se encuentran ubicados en el directorio /tests

para correr los test se ejecuta el comando:
```python -m unittest``` o ```pytest```
