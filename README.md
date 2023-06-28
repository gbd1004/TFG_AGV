[![Maintainability](https://api.codeclimate.com/v1/badges/69966b64b3b9c08413b9/maintainability)](https://codeclimate.com/github/gbd1004/TFG_AGV/maintainability)
[![pylint score](https://github.com/gbd1004/TFG_AGV/actions/workflows/linting.yml/badge.svg)](https://github.com/gbd1004/TFG_AGV/actions/workflows/linting.yml)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

# TFG AGV

En este proyecto se define la creación de un sistema capaz de almacenar datos recibidos de un AGV (Autonomous Guided Vehicle) y de predecir dichos datos. Esto permitirá realizar cosas como mantenimiento predictivo, alertas automáticas, etc.

## Dependencias para la ejecución

Para la ejecución del proyecto, las siguientes herramientas son necesarias:
1. Docker
2. Docker-compose
3. Nvidia-docker (en caso de realizar la predicción de datos con la GPU)

## Instalación

Para la creación de las imágenes de los contenedores de cada servicio, se ejecuta los siguientes comandos desde la carpeta "services":

        docker compose build

## Ejecución

Para la ejecución de los servicios, utilizando la GPU para predecir nuevos datos, se ejecuta el siguiente comando:

        docker compose --profile gpu up --build

Por otra parte, si no puede utilizarse GPU, se ejecuta el siguiente comando:

        docker compose --profile cpu up --build

Se puede ejecutar también el proyecto sin utilizar el servicio de predicción. Para ello basta con no especificar ningún perfil en el comando de ejecución:

        docker compose up --build

Se puede añadir opcionalmente el argumento "-d" para ejecutar los contenedores de manera desacoplada al terminal actual.

## Detener la ejecución.

Para detener la ejecución de los servicios se ejecuta el siguiente comando:

        docker compose down

Opcionalmente, se puede añadir el argumento "-v", lo que eliminará los volúmenes montados por cada contenedor.

## Documentación.

Los archivos de documentación se encuentran en el directorio "memoria/build".