# import libs
import pandas as pd
from sklearn.preprocessing import MinMaxScaler # Normalizador de dados numéricos
from sklearn.cluster import KMeans # KMeans é um metaestimador
from scipy.spatial.distance import cdist #método pronto para avaliar a distância de clusters
import matplotlib.pyplot as plt
import pickle
import math
import numpy as np

# abrir o arquivo de dados
dados = pd.read_csv("../datasets/iris.csv", sep=';')
print(f'Print inicial do dataframe:\n{dados.head(10)}\n')
print(f'Print das colunas do dataframe:\n{dados.columns}\n')
print(f'Dtypes:\n{dados.dtypes}')

# separação de atributos numéricos e atributos categóricos
dados_num = dados.drop(columns=['class']) # dataframe somente com numéricos
dados_cat = dados['class'] # dataframe somente com categóricos

print(f'Print do dataframe categórico:\n{dados_cat.head(10)}\n')
print(f'Print do dataframe numérico:\n{dados_num.head(10)}\n')
print(f'Print das colunas do dataframe:\n{dados_num.columns}\n')

# normalizar dados numéricos com fit()
scaler = MinMaxScaler()
normalizador = scaler.fit(dados_num)

# salvar o modelo normalizador
pickle.dump(normalizador, open('normalizador_iris.pkl', 'wb')) # salva no .pkl no projeto

# normalizar os dados numéricos com fit_transform()
dados_num_norm = normalizador.fit_transform(dados_num)
dados_cat_norm = pd.get_dummies(dados_cat, prefix_sep='_', dtype='int')

# transformar o dadus_num_norm em um dataframe
dados_num_norm = pd.DataFrame(dados_num_norm, columns = dados_num.columns)
print(dados_num_norm.head(10))

# recompor o dataframe com todos os dados
dados_norm = dados_num_norm.join(dados_cat_norm, how='left')
print(dados_norm.head(10))

# hiperparametrizar antes do treinamento
distorcoes = []

#criar um intervalo numérico fechado à esquerda e aberto à direita
K = range(1, 101);

#busca do patamar para atingir a condição de parada
for i in K:
    #treinando iterativamente e aumentando o número de clusters
    cluster_iris = KMeans(n_clusters=i, random_state=42).fit(dados_norm)
    
    #calculo da distorção
    distorcoes.append(
        sum(
            np.min(
                cdist(dados_norm, cluster_iris.cluster_centers_,
                      'euclidean'),
                axis=1
            )/dados_norm.shape[0]
        )
    )
print(distorcoes)

# plotar o gráfico das distorcoes
fig, ax = plt.subplots()
ax.plot(K, distorcoes)
ax.set(xlabel = 'n Clusters', ylabel = 'Distorcoes')
ax.grid()
# plt.show()
plt.savefig('elbow_curve.png', dpi=150, bbox_inches='tight')
plt.close()

#determinar o número ótimo de clusters
x0 = K[0]
y0 = distorcoes[0]
xn = K[-1]
yn = distorcoes[-1]
distancias = []
for i in range(len(distorcoes)):
    x = K[i]
    y = distorcoes[i]
    
    numerador = abs(
        ((yn-y0)*x) - ((xn-x0)*y) + (xn*y0) - (yn*x0)
    )
    
    denominador = math.sqrt(
        ((yn-y0)**2) + ((xn-x0)**2)
    )
    
    distancias.append(numerador/denominador)

numero_clusters_otimo = K[distancias.index(np.max(distancias))]
print('\nNúmero ótimo de clusters: ', numero_clusters_otimo)

#treinar e salvar o modelo de clusters
cluster_iris = KMeans(
    n_clusters=numero_clusters_otimo, random_state=42
).fit(dados_norm)
print(dados_norm.columns)
pickle.dump(cluster_iris, open('cluster_iris.pkl', 'wb'))