import pandas as pd

class Predizer():
    def __init__(self, normalizador_bh, treinador_bh, novo_dado):
        self.normalizador_bh = normalizador_bh
        self.treinador_bh = treinador_bh
        self.novo_dado = novo_dado
    
    def predizer(self):
        dado_preparado = self.preparar()
        dado_predito = self.treinador_bh.predict(dado_preparado)
        return dado_predito
    
    def preparar(self):
        df_novo_dado = pd.DataFrame(
            data=[self.novo_dado],
            columns=self.normalizador_bh.feature_names_in_
        )

        df_novo_dado_norm = pd.DataFrame(
            data=self.normalizador_bh.transform(df_novo_dado),
            columns=df_novo_dado.columns
        )
        
        return df_novo_dado_norm