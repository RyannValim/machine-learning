class Preparar():
    def __init__(self):
        pass
    
    def preparar(self, df):
        colunas = ('CRIM', 'ZN', 'INDUS', 'AGE', 'LSTAT')
        
        for coluna in colunas:
            df[coluna] = df[coluna].fillna(df[coluna].median())
        
        df['CHAS'] = df['CHAS'].fillna(df['CHAS'].mode()[0])
        
        return df
        
    def salvar(self, df):
        df.to_csv('../datasets/boston_housing/HousingDataFillna.csv', index=False)