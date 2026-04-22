import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

class Normalizar():
    def __init__(self, df_iris):
        self.df_iris = df_iris
        self.normalizador_iris = MinMaxScaler()
        self.codificador_ohe_iris = OneHotEncoder(sparse_output=False)
    
    def normalizar(self):
        df_iris_num = self.df_iris.select_dtypes(float)
        self.normalizador_iris.fit(df_iris_num)
        
        df_iris_cat = self.df_iris.select_dtypes(str)
        self.codificador_ohe_iris.fit(df_iris_cat)
        
        df_iris_num_norm = pd.DataFrame(self.normalizador_iris.transform(df_iris_num), columns=df_iris_num.columns)
        
        df_iris_cat_norm = pd.DataFrame(
            self.codificador_ohe_iris.transform(df_iris_cat),
            columns=self.codificador_ohe_iris.get_feature_names_out()
        ).astype(int)
        
        df_iris_normalizado = pd.concat([df_iris_num_norm, df_iris_cat_norm], axis=1)
        return df_iris_normalizado
        
    def salvar(self, df_iris_normalizado):
        df_iris_normalizado.to_csv('../datasets/iris/df_iris_normalizado.csv', index=False, sep=';')
        pickle.dump(self.normalizador_iris, open('../modelos/iris/normalizador_iris.pkl', 'wb'))
        pickle.dump(self.codificador_ohe_iris, open('../modelos/iris/codificador_ohe_iris.pkl', 'wb'))