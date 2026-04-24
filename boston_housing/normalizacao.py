import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class Normalizar():
    def __init__(self):
        self.normalizador_bh = MinMaxScaler()
    
    def normalizar(self, df):
        self.normalizador_bh.fit(df)
        df_norm = pd.DataFrame(self.normalizador_bh.transform(df), columns=df.columns)
        return df_norm
        
    def salvar(self, df):
        df.to_csv('../datasets/boston_housing/HousingDataNorm.csv', index=False)
        pickle.dump(self.normalizador_bh, open('../modelos/boston_housing/minmax_bh.pkl', 'wb'))