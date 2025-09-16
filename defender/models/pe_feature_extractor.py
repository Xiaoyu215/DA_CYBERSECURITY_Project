from .pe_attributes import PEAttributes
import os
import inspect
import pandas as pd
import numpy as np
import pickle

class PEFeatureExtractor():
    # numerical attributes
    NUMERICAL_ATTRIBUTES = ['BaseOfCode', 'BaseOfData', 'Characteristics', 'DllCharacteristics', 
                            'Entropy', 'FileAlignment', 'ImageBase', 'Machine', 'Magic',
                            'NumberOfRvaAndSizes', 'NumberOfSections', 'NumberOfSymbols', 'PE_TYPE',
                            'PointerToSymbolTable', 'Size',
                            'SizeOfCode', 'SizeOfHeaders',
                            'SizeOfImage', 'SizeOfInitializedData', 'SizeOfOptionalHeader',
                            'SizeOfUninitializedData', 'TimeDateStamp']

    # textual attributes
    TEXTUAL_ATTRIBUTES = ['Identify', 'ImportedDlls', 'ImportedSymbols']
    
    
    # initialize extracting attributes from PE file using an previous trained extractor and scaler
    def __init__(self, file, extractor, scaler, strings=False):
        # initialize pe attribute extractor
        pe_a = PEAttributes(file, strings)
        # get attributes values and names
        values, attributes = pe_a.attr_values()
        print(attributes)
        # create dataframe with obtained values
        self.attributes = pd.DataFrame([values], columns=attributes)
        # load extractor
        self.extractor = pickle.load(open(extractor, 'rb'))
        # load scaler
        self.scaler = pickle.load(open(scaler, 'rb'))
        
    # extract features from PE file using TF-IDF and normalize using MinMax
    def extract_features(self):
        self.features = self.attributes[self.NUMERICAL_ATTRIBUTES].values
        # extract features from each textual attribute
        for a in self.TEXTUAL_ATTRIBUTES:
            # extract features from current attribute
            train_texts = self.extractor.transform(self.attributes[a])
            # concatenate with numerical attributes
            self.features = np.concatenate((self.features, train_texts.toarray()), axis=1)
        # normalize
        self.features = self.scaler.transform(self.features)
        # return features
        return(self.features)        