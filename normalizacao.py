import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

class Normalizar():
    def __init__(self, df):
        self.df = df
        self.minmax_norm = MinMaxScaler()
        self.codificador_ohe = OneHotEncoder(sparse_output=False)
    
    def normalizar(self):
        df_num = self.df.select_dtypes(float)
        self.minmax_norm.fit(df_num)
        
        df_cat = self.df.select_dtypes(str)
        self.codificador_ohe.fit(df_cat)
        
        df_num_norm = pd.DataFrame(self.minmax_norm.transform(df_num), columns=df_num.columns)
        
        df_cat_norm = pd.DataFrame(
            self.codificador_ohe.transform(df_cat),
            columns=self.codificador_ohe.get_feature_names_out()
        ).astype(int)
        
        df_normalizado = pd.concat([df_num_norm, df_cat_norm], axis=1)
        return df_normalizado
        
    def salvar(self, df_normalizado):
        df_normalizado.to_csv('./datasets/df_normalizado.csv', index=False, sep=';')
        pickle.dump(self.minmax_norm, open('./modelos/minmax_norm.pkl', 'wb'))
        pickle.dump(self.codificador_ohe, open('./modelos/codificador_ohe.pkl', 'wb'))