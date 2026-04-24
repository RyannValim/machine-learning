class Preparar():    
    def preparar(self, df):
        colunas = ('CRIM', 'ZN', 'INDUS', 'AGE', 'LSTAT')
        df['CHAS'] = df['CHAS'].fillna(df['CHAS'].mode()[0])
        
        for coluna in colunas:
            df[coluna] = df[coluna].fillna(df[coluna].median())        
        
        return df
        
    def salvar(self, df):
        df.to_csv('../datasets/boston_housing/HousingDataFillna.csv', index=False)