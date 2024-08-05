
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

class DataPreprocessor:

    def __init__(self, data = None, missing_data = None, levels = None, orders = None, normalization = None):

        self.data = data
        self.missing_data = missing_data
        self.levels =  levels
        self.orders = orders
        self.normalization = normalization
        self.processed_data = None
        self.encoders = None
        self.X = None
        self.G = None
    

    def process_missing_data(self, data):
        processed_data = data.copy()

        if self.missing_data == "fill":
            
            num_imputer = SimpleImputer(strategy="mean")
            cat_imputer = SimpleImputer(strategy="most_frequent")

            
            for col in processed_data.select_dtypes(include='number').columns:
                processed_data[col] = num_imputer.fit_transform(processed_data[[col]])
            
            for col in processed_data.select_dtypes(include='object').columns:
                processed_data[col] = cat_imputer.fit_transform(processed_data[[col]])

        elif self.missing_data == "drop":
            processed_data.dropna(axis=0, inplace=True)

        return pd.DataFrame(processed_data)
    

    def process_encoding(self, data):
        columns = data.columns
        processed_data = data.copy().to_numpy()
        G = []
        
        for i, col in enumerate(columns):
            print(i)
            print(col)
            
            print('-------')
            encoder = None
            if self.encoders[i] is not None:
                encoder = self.encoders[i]
        
            encoded_col = processed_data[:,i].reshape(-1,1)

            if isinstance(encoder, OneHotEncoder):
                encoded_col = encoder.fit_transform(encoded_col)
            if isinstance(encoder, OrdinalEncoder):
                print(self.orders[col])
                encoder.categories_=self.orders[col]
                print(encoder.categories_)
                encoded_col = encoder.fit_transform(encoded_col)

            G.append(encoded_col)
            if isinstance(encoder, OneHotEncoder):
                encoded_col = encoded_col
            if i == 0:
                encoded_data = encoded_col
            else:
                encoded_data = np.hstack((encoded_data, encoded_col))
        self.G = G
        self.X = np.hstack(G)
        return pd.DataFrame(encoded_data)
    
    def process_data(self):

        processed_data = self.data.copy()
        self.make_encoders()
        if self.missing_data is not None:
            processed_data = self.process_missing_data(processed_data)
        if self.levels is not None:
            processed_data = self.process_encoding(processed_data)
        if self.normalization is not  None:
            processed_data = self.process_normalization(processed_data)
        
        self.processed_data = pd.DataFrame(processed_data)

    def process_normalization(self, data):
        scaler = StandardScaler()
        pass

    
    def make_encoders(self):
        encoders = []

        for col in self.data.columns:

            if self.levels[col] == 'nominal':
                encoders.append(OneHotEncoder(sparse_output=False))
            elif self.levels[col] == 'ordinal':
                encoders.append(OrdinalEncoder())
            else:
                encoders.append(None)
            
            self.encoders = encoders


        





