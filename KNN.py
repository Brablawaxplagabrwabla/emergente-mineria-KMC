# Primero se importan las librerías necesarias
# pandas es una librería de análisis de datos
# mientras que sklearn es una librería de machine learning
from sklearn import datasets
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn. neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Primero se carga el conjunto de datos, se trata del iris dataset
# el cual fue empleado por Ronald Fisher para probar técnicas
# de análisis discriminante lineal
iris = datasets.load_iris()
X = pd.DataFrame(iris['data'], columns=iris['feature_names'])

# Cargamos los resultados esperados para poder realizar el
# entrenamiento supervisado

y = iris['target']
X.head()

# Para realizar el entrenamiento es conveniente realizar una
# prenormalización de la data
escalador = preprocessing.MinMaxScaler()
X = escalador.fit_transform(X)

# Ahora dividimos el dataset en un conjunto de entrenamiento
# y otro conjunto de prueba, en una relación 70 - 30
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(X, y, random_state=1)

# Aquí aplicamos Análisis de Componente Principal, puesto que
# el conjunto de datos es 4-dimensional, pero el 97% de la 
# varianza puede explicarse en un modelo 2D, esto se mostrará
# luego
modelo_pca = PCA(n_components=2)
modelo_pca.fit(X_entrenamiento)
# Aplicamos la transformación para obtener dos vectores de datos
# que son proyecciones ortogonales de los vectores 4D originales
# en el plano, tanto para el conjunto de entrenamiento, como
# para el conjunto de prueba
X_entrenamiento = modelo_pca.transform(X_entrenamiento)
X_prueba = modelo_pca.transform(X_prueba)

# Varianza 2D
print(modelo_pca.explained_variance_ratio_)
print(sum(modelo_pca.explained_variance_ratio_))

X_entrenamiento[:5]

k = 5

# Aplicamos el modelo de los k vecinos al conjunto de entre-
# miento
modelo_knn = KNeighborsClassifier(n_neighbors=k)
modelo_knn.fit(X_entrenamiento, y_entrenamiento)

# Utilizamos el modelo entrenado para predecir con el conjunto
# de prueba
y_predicha = modelo_knn.predict(X_prueba)
y_predicha

# Ahora procedemos a realizar una gráfica de la partición de
# realizada sobre los conjuntos 2-dimensionales producidos por
# análisis de componente principal

matplotlib.style.use('ggplot')

figura = plt.figure(figsize=(8, 6))
ax = figura.add_subplot(111)
padding = 0.1
resolucion = 0.1

colores = {0: 'violet', 1: 'indigo', 2: 'palegreen'}
x_min, x_max = X_entrenamiento[:, 0].min(), X_entrenamiento[:, 0].max()
y_min, y_max = X_entrenamiento[:, 1].min(), X_entrenamiento[:, 1].max()
rango_x = x_max - x_min
rango_y = y_max - y_min
x_min -= rango_x * padding
y_min -= rango_y * padding
x_max += rango_x * padding
y_max += rango_y * padding

xx, yy = np.meshgrid(np.arange(x_min, x_max, resolucion),
	np.arange(y_min, y_max, resolucion))

Z = modelo_knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.PRGn)
plt.axis('tight')

# Especie 0 representa setosa, 1 representa versicolor y 2
# representa virginica
for etiqueta in np.unique(y_prueba):
	indices = np.where(y_prueba == etiqueta)
	plt.scatter(X_prueba[indices, 0], X_prueba[indices, 1], c=colores[etiqueta], alpha=0.8, label='Especie {}'.format(etiqueta))
plt.legend(loc='lower right')
plt.title('K = {}'.format(k))
plt.show()