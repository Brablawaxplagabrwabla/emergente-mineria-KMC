# Primero se importan las librerías necesarias
# pandas es una librería de análisis de datos
# mientras que sklearn es una librería de machine learning
import pandas as pd
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Primero se carga el conjunto de datos, se trata del iris dataset
# el cual fue empleado por Ronald Fisher para probar técnicas
# de análisis discriminante lineal
iris = datasets.load_iris()
X = pd.DataFrame(iris['data'], columns=iris['feature_names'])

# Aquí aplicamos Análisis de Componente Principal, puesto que
# el conjunto de datos es 4-dimensional, pero el 97% de la 
# varianza puede explicarse en un modelo 2D, esto se mostrará
# luego
pca = PCA(n_components=2, whiten=True).fit(X)

# Aplicamos la transformación para obtener dos vectores de datos
# que son proyecciones ortogonales de los vectores 4D originales
# en el plano
X = pca.transform(X)

# Varianza 2D
print(pca.explained_variance_ratio_)
print(sum(pca.explained_variance_ratio_))

# Ahora debo formatear los datos de manera que puedan ser usados
# por la librería que aplicará el k-means clustering
x = []
y = []

for i in range(0, len(X)):
	x.append(X[i][0])
	y.append(X[i][1])

# Aplicamos el método del clústering de las k medias, puesto
# que en el dataset hay datos de tres especias distintas, se
# desea encontrar tres clústers, y por ello se emplea un valor
# de k = 3

kmedias = KMeans(n_clusters=3)
# Ahora le pedimos que defina los tres clústers de datos
kmedias.fit(X)

# Le pedimos ahora que clasifique el conjunto de datos de 
# acuerdo a los clústers que halló
clasificacion = kmedias.predict(X)
# Hallamos los centroides alrededor de los cuales se forman
# los clústers, para poder graficarlos
centroides = kmedias.cluster_centers_

# Ahora se realizan algunas configuraciones respecto al
# gráfico que se desea realizar
figura = plt.figure(figsize=(5, 5))
colmap = {1: 'r', 2: 'g', 3: 'b'}
colores = ''
for i in range(0, len(clasificacion)):
	if (clasificacion[i] == 0):
		colores += 'r'
	if (clasificacion[i] == 1):
		colores += 'g'
	if (clasificacion[i] == 2):
		colores += 'b'

# En la gráfica, el verde representa una clasificación de la
# flor como de especie setosa, el rojo representa la especie
# versicolor y el azul una clasificación como virginica
plt.scatter(x, y, 20, colores, alpha=0.8, edgecolor='k')
for idx, centroide in enumerate(centroides):
	plt.scatter(*centroide, color=colmap[idx+1])
plt.show()

# Cálculo de errores

# Un valor de 0 en las predicciones del método KMC significa 
# setosa, 1 significa versicolor y 2 significa virginica

error = 0
print(iris.target)
print(clasificacion)

for i in range(0, len(clasificacion)):
    if (iris.target[i] == 0 and clasificacion[i] != 0):
        error += 1
    if (iris.target[i] == 1 and clasificacion[i] != 1):
        error += 1
    if (iris.target[i] == 2 and clasificacion[i] != 2):
        error += 1

errorPorcentual = (error * 100)/150
print(errorPorcentual)