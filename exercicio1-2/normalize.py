import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder


class Normalize:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.encoder = LabelEncoder()
        self.ohe = OneHotEncoder(sparse_output=False)
        self.ohe_columns = []

    def minmax(self, df, columns):
        df[columns] = self.scaler.fit_transform(df[columns])
        return df

    def inverse_minmax(self, df, columns):
        df[columns] = self.scaler.inverse_transform(df[columns])
        return df

    def label(self, df, column):
        df[column] = self.encoder.fit_transform(df[column])
        return df

    def inverse_label(self, df, column):
        df[column] = self.encoder.inverse_transform(df[column])
        return df

    def onehot(self, df, column):
        expanded = self.ohe.fit_transform(df[[column]])
        self.ohe_columns = self.ohe.get_feature_names_out([column]).tolist()
        df_ohe = pd.DataFrame(expanded, columns=self.ohe_columns, index=df.index)
        df = df.drop(columns=[column])
        df = pd.concat([df, df_ohe], axis=1)
        return df

    def inverse_onehot(self, df, original_column):
        reversed_values = self.ohe.inverse_transform(df[self.ohe_columns])
        df[original_column] = reversed_values[:, 0]
        df = df.drop(columns=self.ohe_columns)
        return df
