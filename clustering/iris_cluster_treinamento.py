# import libs
import pandas as pd
from sklearn.preprocessing import MinMaxScaler # Normalizador de dados numéricos
from sklearn.cluster import KMeans # KMeans é um metaestimador
import pickle

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
