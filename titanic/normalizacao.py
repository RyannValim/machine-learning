import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

class Normalizador():
    def __init__(self, df):
        self.df = df
        self.minmax_norm = MinMaxScaler()
        self.ohe_encoder = OneHotEncoder(sparse_output=False)
    
    def normalizar(self):
        # separação dos dados numéricos x categóricos
        cols_num = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
        cols_cat = ['Sex', 'Embarked']
        
        dados_num = self.df[cols_num]
        dados_cat = self.df[cols_cat]
        
        # normalização dos dados
        self.minmax_norm.fit(dados_num)
        self.ohe_encoder.fit(dados_cat)
        
        array_cat_norm = self.ohe_encoder.get_feature_names_out(cols_cat)
        
        df_num_norm = pd.DataFrame(
            data=self.minmax_norm.transform(dados_num),
            columns=cols_num
        )
        
        df_cat_norm = pd.DataFrame(
            data=self.ohe_encoder.transform(dados_cat),
            columns=array_cat_norm
        )

        return pd.concat([df_num_norm, df_cat_norm], axis=1)
    
    def salvar(self, df):
        df.to_csv('../datasets/titanic/TitanicNormalizado.csv', index=False)
        pickle.dump(self.minmax_norm, open('../modelos/titanic/minmax_titanic.pkl', 'wb'))
        pickle.dump(self.ohe_encoder, open('../modelos/titanic/ohe_encoder_titanic.pkl', 'wb'))