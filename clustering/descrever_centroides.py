import pickle
import pandas as pd

#salvando os nomes das colunas
nomes_colunas = ['sepal_length',
                 'sepal_width',
                 'petal_length',
                 'petal_width',
                 'Iris-setosa',
                 'Iris-versicolor',
                 'Iris-virginica']
print(f'Nome das colunas:\n{nomes_colunas}\n')

#abrir o modelo treinado
cluster_iris = pickle.load(open('./cluster_iris.pkl', 'rb'))

#imprimir os valores dos centróides
# print(f'Centróides:\n{cluster_iris.cluster_centers_}\n')

#converter os centróides em dataframe
centroides = pd.DataFrame(cluster_iris.cluster_centers_, columns=nomes_colunas)
print(f'Centróides em DataFrame:\n{centroides}\n')

#segmentar o dataframe em colunas numéricas e colunas categóricas
dados_num_norm = centroides.drop(columns=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
dados_cat_norm = centroides[['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']]

print(f'Dados numéricos normalizados:\n{dados_num_norm}\n')
print(f'Dados categóricos normalizados:\n{dados_cat_norm}\n')

#desnormalizar as colunas numéricas
##carregar o normalizador que foi salvo durante o preprocessamento
normalizador = pickle.load(open('./normalizador_iris.pkl', 'rb'))
dados_num = normalizador.inverse_transform(dados_num_norm)

print(f'Dados numéricos de volta para vetor:\n{dados_num}\n')

#atenção: após desnormalizar os dados numéricos, teremos uma matriz do numpy
##será necessário recriar o dataframe
dados_num = pd.DataFrame(dados_num, columns=dados_num_norm.columns)
print(dados_num)