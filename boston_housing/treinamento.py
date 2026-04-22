import pickle
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

class Treinar():
    def __init__(self):
        self.treinador = KMeans()
    
    def treinar(self, df):
        k_otimo = self.calc_elbow(df)
        self.treinador = KMeans(n_clusters=k_otimo, random_state=42).fit(df)
        return self.treinador
    
    def calc_elbow(self, df):
        distorcoes = []
        clusters = range(1, 11)
        for c in clusters:
            self.treinador = KMeans(n_clusters=c, random_state=42).fit(df)
            distorcoes.append(
                sum(np.min(cdist(
                    df, self.treinador.cluster_centers_, 'euclidean'),
                           axis=1) / df.shape[0])
            )
        
        distancias = []
        for i in range(len(distorcoes)):
            x = clusters[i]
            y = distorcoes[i]
            
            x0 = clusters[0]
            y0 = distorcoes[0]
            xn = clusters[-1]
            yn = distorcoes[-1]

            distancias.append(abs(((
                (yn - y0)*x) - (xn - x0)*y) + (xn * y0) - (yn * x0)
            ) / sqrt((yn - y0)**2 + (xn - x0)**2))
        
        plt.plot(clusters, distorcoes)
        plt.savefig('../plotagens/boston_housing/elbow_curve_bh.png')
        plt.close()
        
        return clusters[distancias.index(np.max(distancias))]
        
    def salvar(self, treinador):
        pickle.dump(treinador, open('../modelos/boston_housing/treinador_bh.pkl', 'wb'))