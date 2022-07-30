# model_lpm
Creación de modelos de regresión lineal con el mínimo de error en latidos por minuto respecto a las mediciones reales

### DBSCAN
Se usa BDSCAN de machine learning para crear clusters de la información leida:
* en un primer paso se usa DBSCAN con el cálculo automáyico de eps
* en segundo término se usa DBSCAN con eps=3 de acuerdo a la sugerencia de la literatura científica relacionada con el número de latidos por minuto. (ver prefijo B en las notaciones)

### BANDAS
Posteriormente de cada cluster creado con DBSCAN se crean bandas de mediciones en un rango de 6 latidos por minuto, esto para entrar en el límite de 3 lpm por encima y por debajo de la gráfica de mínimos cuadrados.

### Graficas SVG
Las gráficas que se crean están en formato SVG y corresponden a cada cluster y las bandas correspondientes de cada cluster.

### Errores
Se describe una serie de errores que ayudan al usario a definir sobre que criterio debe elejir el modelo que desea usar.

# USO
Desde MSDOS escribir el comando    *eq.exe < archivo.csv > archivo_salida.txt*

donde:
* archivo.csv         tiene los valores de y (variable dependiente), x (variable independiente) e id (identificador de la persona)
* archivo_salida.txt  respuestas textuales de la aplicación que describe el análisis realizado.  

