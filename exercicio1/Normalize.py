# class Normalize():
#     def __init__(self):
#         self.minmax_scaler = MinMaxScaler()
#         self.label_encoder = LabelEncoder()

#     def fit_minmax(self, df: pd.DataFrame, columns: list):
#         self.minmax_columns = columns
#         self.minmax_scaler.fit(df[columns])
    
#     def transform_minmax(self, df: pd.DataFrame) -> pd.DataFrame:
#         df[self.minmax_columns] = self.minmax_scaler.transform(df[self.minmax_columns])
#         return df

#     def inverse_minmax(self, df: pd.DataFrame) -> pd.DataFrame:
#         df[self.minmax_columns] = self.minmax_scaler.inverse_transform(df[self.minmax_columns])
#         return df
    
#     def fit_label(self, df: pd.DataFrame, columns: list):
#         df[self.label_encoder] = self.label_encoder.fit()
#         pass

#     def transform_label(self, df: pd.DataFrame) -> pd.DataFrame:
#         df[self.label_encoder] = self.label_encoder.transform()
#         pass

#     def inverse_label(self, df: pd.DataFrame) -> pd.DataFrame:
#         df[self.label_encoder] = self.label_encoder.inverse_transform()
#         pass