""" Exercício 1
Implementar uma classe capaz de normalizar dados conforme os métodos
citados em sala:

- MinMax Scaler;
- Label Encoding;
- One-Hot Encodig;
- A classe (em Python) deve ser reaproveitável;
- Deve-se implementar os métodos de reversão.

Utilizar o arquivo "dados_normalizar.csv" para testar o código.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder

df = pd.read_csv("../data/dados_normalizar.csv", sep=';', decimal=',')
print(df.head())
print(df.dtypes)

encoder = LabelEncoder()
scaler = MinMaxScaler()
one_hot_encoder = OneHotEncoder(sparse_output=False)

df_normalizado = scaler.fit_transform(df[['idade', 'altura', 'Peso']])
df[['idade', 'altura', 'Peso']] = df_normalizado
print(df.head())

colunas_normalizadas = encoder.fit_transform(df['sexo'])
df['sexo'] = colunas_normalizadas
print(df.head())

reversao_df = scaler.inverse_transform(df[['idade', 'altura', 'Peso']])
df[['idade', 'altura', 'Peso']] = reversao_df
print(df.head())

reversao_colunas = encoder.inverse_transform(df['sexo'])
df['sexo'] = reversao_colunas
print(df.head())

# ONE-HOT ENCODER
