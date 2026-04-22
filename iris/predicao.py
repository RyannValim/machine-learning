import pandas as pd

class Predizer:
    def __init__(self, normalizador_iris, codificador_ohe_iris, treinador_iris):
        self.normalizador_iris = normalizador_iris
        self.codificador_ohe_iris = codificador_ohe_iris
        self.treinador_iris = treinador_iris

    def preparar(self, nova_flor):
        dados_nova_flor = pd.DataFrame(
            [nova_flor],
            columns=self.normalizador_iris.feature_names_in_
        )
        zeros = [0] * len(self.codificador_ohe_iris.get_feature_names_out())
        colunas_ohe = pd.DataFrame([zeros], columns=self.codificador_ohe_iris.get_feature_names_out())
        nova_flor_norm = self.normalizador_iris.transform(dados_nova_flor)
        df_nova_flor_norm = pd.DataFrame(nova_flor_norm, columns=dados_nova_flor.columns)
        df_nova_flor = pd.concat([df_nova_flor_norm, colunas_ohe], axis=1)
        return df_nova_flor

    def predizer(self, df_nova_flor):
        cluster_nova_flor = self.treinador_iris.predict(df_nova_flor)
        return cluster_nova_flor