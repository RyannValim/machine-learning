import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class Normalizar():
    def __init__(self):
        self.normalizador = MinMaxScaler()
    
    def normalizar(self, df):
        self.normalizador.fit(df)
        df_norm = pd.DataFrame(self.normalizador.transform(df), columns=df.columns)
        return df_norm
        
    def salvar(self, df, modelo):
        df.to_csv('../datasets/boston_housing/HousingDataNorm.csv', index=False)
        pickle.dump(modelo, open('../modelos/boston_housing/minmax_bh.pkl', 'wb'))