# model_lpm
Creación de modelos de regresión lineal con el mínimo de error en latidos por minuto (lpm) respecto a las mediciones reales.
Usar la ###**carpeta todo**###

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
Desde MSDOS escribir el comando:    *___eq.exe < archivo.csv > salida.txt___*

donde:
* archivo.csv         tiene los valores de y (variable dependiente), x (variable independiente) e id (identificador de la persona)
* salida.txt  respuestas textuales de la aplicación que describe el análisis realizado.  

Ejemplo:
![image](https://github.com/user-attachments/assets/8adc1d43-a2ea-46b4-b5cb-3c82de926b8a)


# Usar después juntarGG.exe
### para ver más fásilmente como se distribuye la intensidad usando frecuencia cardiaca
![plot](https://github.com/user-attachments/assets/56d7e62a-0a61-4a38-9c67-fe902c121eaf)
![todo](https://github.com/user-attachments/assets/3dc4cd55-9fb5-4a9c-b3f1-de6bb2e6932b)

