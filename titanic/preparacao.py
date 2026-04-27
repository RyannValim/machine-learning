class Preparador():
    def __init__(self, df):
        self.df = df
    
    def preparar(self):
        # tratando das colunas que serão arredondadas com a mediana
        colunas_mediana = ['Age', 'SibSp', 'Parch']
        for coluna in colunas_mediana:
            self.df[coluna] = self.df[coluna].fillna(self.df[coluna].median())
        
        # tratando das colunas que serão arredondadas com a media
        self.df['Fare'] = self.df['Fare'].fillna(self.df['Fare'].mean())
        
        # tratando das colunas que serão arredondadas com a moda
        self.df['Embarked'] = self.df['Embarked'].fillna(self.df['Embarked'].mode()[0])
        
        # drop colunas
        colunas_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
        
        # retorno
        return self.df.drop(columns=colunas_drop)
    
    def salvar(self, df):
        df.to_csv('../datasets/titanic/TitanicPreparado.csv', index=False)