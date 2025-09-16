# this file is a test to create an attribute extractor to generate usefull features :D

import os
import re
import lief
import pefile
import peutils
import math
import numpy as np
import pandas as pd
import pickle

class PEAttributeExtractor():

    libraries = ""
    functions = ""
    exports = ""

    # initialize extractor
    def __init__(self, bytez):
        # save bytes
        self.bytez = bytez
        # save pe
        # self.pe = pefile.PE(data=bytez, fast_load=True)
        # parse using lief
        self.lief_binary = lief.PE.parse(list(bytez))
        # attributes
        self.attributes = {}

    # extract string metadata
    def extract_string_metadata(self):
        # occurances of the string 'C:\'.  Not actually extracting the path
        paths = re.compile(b'c:\\\\', re.IGNORECASE)
        # occurances of http:// or https://.  Not actually extracting the URLs
        urls = re.compile(b'https?://', re.IGNORECASE)
        # occurances of the string prefix HKEY_.  No actually extracting registry names
        registry = re.compile(b'HKEY_')
        # crude evidence of an MZ header (dropper?) somewhere in the byte stream
        mz = re.compile(b'MZ')
        return {
            'string_paths': len(paths.findall(self.bytez)),
            'string_urls': len(urls.findall(self.bytez)),
            'string_registry': len(registry.findall(self.bytez)),
            'string_MZ': len(mz.findall(self.bytez))
        }

    # extract entropy
    def extract_entropy(self):
        if not self.bytez:
            return 0
        entropy=0
        for x in range(256):
            p_x = float(self.bytez.count(bytes(x)))/len(self.bytez)
            if p_x>0:
                entropy += - p_x*math.log(p_x, 2)
        return entropy

    # extract identify
    def extract_identify(self):
        # Load PE Signature Database & Sample PE
        # db_path = os.path.dirname(__file__) + "defender/models/peid_userdb_uft8.txt"
        # sigs=peutils.SignatureDatabase(db_path) #data=open(db_path, 'r').read())
        # # Match PE against signature database
        # matches=sigs.match_all(self.pe, ep_only=True)
        # m = []
        # if matches:
        #     for l in matches:
        #         for i in l:
        #             m.append(str(i))
        #     return " ".join(m)  
        # else:
        return ""
    
    # extract attributes
    def extract(self):

        # get general info
        self.attributes.update({
            "size": len(self.bytez), 
            # EMBER only
            "virtual_size": self.lief_binary.virtual_size,
            # EMBER only
            "has_debug": int(self.lief_binary.has_debug), 
            # EMBER only
            "imports": len(self.lief_binary.imports),
            # EMBER only
            "exports": len(self.lief_binary.exported_functions),
            # EMBER only
            "has_relocations": int(self.lief_binary.has_relocations),
            # EMBER only
            "has_resources": int(self.lief_binary.has_resources),
            # EMBER only
            "has_signature": int(self.lief_binary.has_signature),
            # EMBER only
            "has_tls": int(self.lief_binary.has_tls),
            # EMBER only
            "symbols": len(self.lief_binary.symbols),
        })

        # get header info
        self.attributes.update({
            "timestamp": self.lief_binary.header.time_date_stamps,
            # TODO: do we transform MACHINE into categorical feature instead of int?
            "machine": str(self.lief_binary.header.machine),
            # TODO: NFS only
            "numberof_sections": self.lief_binary.header.numberof_sections,
            # TODO: NFS only
            "numberof_symbols": self.lief_binary.header.numberof_symbols,
            # TODO: NFS only
            "pointerto_symbol_table": self.lief_binary.header.pointerto_symbol_table,
            # TODO: NFS only
            "sizeof_optional_header": self.lief_binary.header.sizeof_optional_header,
            # TODO: NFS only
            "characteristics": int(self.lief_binary.header.characteristics),
            "characteristics_list": " ".join([str(c).replace("HEADER_CHARACTERISTICS.","") for c in self.lief_binary.header.characteristics_list])
        })

        try:
            baseof_data = self.lief_binary.optional_header.baseof_data
        except:
            baseof_data = 0

        # get optional header
        self.attributes.update({
            # TODO: NFS only
            "baseof_code": self.lief_binary.optional_header.baseof_code,
            # TODO: NFS only
            "baseof_data": baseof_data,
            # TODO: Ember uses a dll_characteristics list
            "dll_characteristics": self.lief_binary.optional_header.dll_characteristics,
            "dll_characteristics_list": " ".join([str(d).replace("DLL_CHARACTERISTICS.", "") for d in self.lief_binary.optional_header.dll_characteristics_lists]),
            # TODO: NFS only
            "file_alignment": self.lief_binary.optional_header.file_alignment,
            # TODO: NFS only
            "imagebase": self.lief_binary.optional_header.imagebase,
            "magic": str(self.lief_binary.optional_header.magic).replace("PE_TYPE.",""),
            # TODO: NFS only - using pefile
            # "PE_TYPE": self.pe.PE_TYPE,
            "PE_TYPE": int(self.lief_binary.optional_header.magic),
            # EMBER only
            "major_image_version": self.lief_binary.optional_header.major_image_version,
            # EMBER only
            "minor_image_version": self.lief_binary.optional_header.minor_image_version,
            # EMBER only
            "major_linker_version": self.lief_binary.optional_header.major_linker_version,
            # EMBER only
            "minor_linker_version": self.lief_binary.optional_header.minor_linker_version,
            # EMBER only
            "major_operating_system_version": self.lief_binary.optional_header.major_operating_system_version,
            # EMBER only
            "minor_operating_system_version": self.lief_binary.optional_header.minor_operating_system_version,
            # EMBER only
            "major_subsystem_version": self.lief_binary.optional_header.major_subsystem_version,
            # EMBER only
            "minor_subsystem_version": self.lief_binary.optional_header.minor_subsystem_version,
            # TODO: NFS only
            "numberof_rva_and_size": self.lief_binary.optional_header.numberof_rva_and_size,
            "sizeof_code": self.lief_binary.optional_header.sizeof_code,
            "sizeof_headers": self.lief_binary.optional_header.sizeof_headers,
            # EMBER only
            "sizeof_heap_commit": self.lief_binary.optional_header.sizeof_heap_commit,
            # TODO: NFS only
            "sizeof_image": self.lief_binary.optional_header.sizeof_image,
            # TODO: NFS only
            "sizeof_initialized_data": self.lief_binary.optional_header.sizeof_initialized_data,
            # TODO: NFS only
            "sizeof_uninitialized_data": self.lief_binary.optional_header.sizeof_uninitialized_data,
            # EMBER only
            "subsystem": str(self.lief_binary.optional_header.subsystem).replace("SUBSYSTEM.","")
        })

        # get entropy
        self.attributes.update({
            # TODO: NFS only
            "entropy": self.extract_entropy()
        })

        # get string metadata
        # EMBER only
        self.attributes.update(self.extract_string_metadata())
        
        # get imported libraries and functions
        if self.lief_binary.has_imports:
            self.libraries = " ".join([l for l in self.lief_binary.libraries])
            self.functions = " ".join([f.name for f in self.lief_binary.imported_functions])
        self.attributes.update({"functions": self.functions, "libraries": self.libraries})

        # get exports
        if self.lief_binary.has_exports:
            self.exports = " ".join([f.name for f in self.lief_binary.exported_functions])
        self.attributes.update({"exports_list": self.exports})
        print(self.exports)

        # get identify
        self.attributes.update({"identify": self.extract_identify()})

        return(self.attributes)

class PEFeatureExtractor():
    # numerical attributes
    NUMERICAL_ATTRIBUTES = ['baseof_code', 'baseof_data', 'characteristics', 'dll_characteristics', 
                            'entropy', 'file_alignment', 'imagebase', 'machine', 'magic',
                            'numberof_rva_and_size', 'numberof_sections', 'numberof_symbols', 'PE_TYPE',
                            'pointerto_symbol_table', 'size', 'sizeof_code', 'sizeof_headers',
                            'sizeof_image', 'sizeof_initialized_data', 'sizeof_optional_header',
                            'sizeof_uninitialized_data', 'timestamp']

    # textual attributes
    TEXTUAL_ATTRIBUTES = ['identify', 'libraries', 'functions']
    
    # initialize extracting attributes from PE file using an previous trained extractor and scaler
    def __init__(self, file, extractor, scaler, strings=False):
        # initialize pe attribute extractor
        pe_a = PEAttributeExtractor(file)
        # get attributes values and names
        atts = pe_a.extract()
        print(atts)
        # create dataframe with obtained values
        self.attributes = pd.DataFrame([atts.values()], columns=atts.keys())
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


input_file = "../MLSEC_2019_samples_and_variants/050"
# input_file = "../MLSEC_2019_samples_and_variants/025_u008_s016"

# input_file = "../MLSEC_2019_samples_and_variants/013_u008_s092"
 # feature extractor used
nfs_extractor = os.path.dirname(__file__) + 'defender/models/nfs_behemot/nfs_extractor_tfidf.pkl'
# scaler used
nfs_scaler = os.path.dirname(__file__) + 'defender/models/nfs_behemot/nfs_scaler_minmax.pkl'

CLASSIFIER = "randomforest_100"
# base name of classifiers
nfs_clf_base_name = os.path.dirname(__file__) + "defender/models/nfs_behemot/nfs_classifier_{}.pkl".format(CLASSIFIER)

# open file
file = open(input_file, "rb")
# read file and get its bytes
bytez = file.read()

fe = PEFeatureExtractor(bytez, nfs_extractor, nfs_scaler)

# print(fe.extract_features())

# # initialize attribute extraction
# att_extraction = PEAttributeExtractor(bytez)
# # extraact attributes
# atts = att_extraction.extract()

# print(atts.values())

# atts = pd.DataFrame([atts.values()], columns=atts.keys())
# print(atts[COLUMNS])