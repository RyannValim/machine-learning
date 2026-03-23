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

from normalize import Normalize
import pandas as pd

#Leitura da CSV
df = pd.read_csv("../data/dados_normalizar.csv", sep=';', decimal=',')
n = Normalize()

#MinMaxScaler
df = n.minmax(df, ['idade', 'altura', 'Peso'])
print(f'MINMAX:\n{df.head()}\n')

#MinMaxScaler inverso
df = n.inverse_minmax(df, ['idade', 'altura', 'Peso'])
print(f'INVERSE MINMAX:\n{df.head()}\n')

#LabelEncoder
df = n.label(df, 'sexo')
print(f'LABEL:\n{df.head()}\n')

#LabelEncoder inverso
df = n.inverse_label(df, 'sexo')
print(f'INVERSE LABEL:\n{df.head()}\n')

#One-Hot Encoder 
df = n.onehot(df, 'sexo')
print(f'ONE-HOT ENCODER:\n{df.head()}\n')

#One-Hot Encoder inverso
df = n.inverse_onehot(df, 'sexo')
print(f'INVERSE ONE-HOT ENCODER:\n{df.head()}\n')