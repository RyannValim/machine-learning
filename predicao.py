import pandas as pd

class Predizer:
    def __init__(self, minmax_norm, codificador_ohe, cluster_treinador):
        self.minmax_norm = minmax_norm
        self.codificador_ohe = codificador_ohe
        self.cluster_treinador = cluster_treinador

    def preparar(self, novo_dado):
        dados_novo_dado = pd.DataFrame(
            [novo_dado],
            columns=self.minmax_norm.feature_names_in_
        )
        zeros = [0] * len(self.codificador_ohe.get_feature_names_out())
        colunas_ohe = pd.DataFrame([zeros], columns=self.codificador_ohe.get_feature_names_out())
        novo_dado_norm = self.minmax_norm.transform(dados_novo_dado)
        df_novo_dado_norm = pd.DataFrame(novo_dado_norm, columns=dados_novo_dado.columns)
        df_novo_dado = pd.concat([df_novo_dado_norm, colunas_ohe], axis=1)
        return df_novo_dado

    def predizer(self, df_novo_dado):
        cluster_novo_dado = self.cluster_treinador.predict(df_novo_dado)
        return cluster_novo_dado