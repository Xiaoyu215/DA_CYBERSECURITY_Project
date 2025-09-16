import pefile
import peutils
# import commands
import hashlib
import curses.ascii
import math
import datetime
import os
import inspect

# extract attributes from a single PE file
class PEAttributes:
    def __init__(self,bytez,name="",strings=False):
        # ignored features
        self.ignored_features = ["getDIRECTORY_ENTRY_IMPORT", "getDOS_HEADER", "getFILE_HEADER", "getOPTIONAL_HEADER"]
        # save file name
        self.name=name
        # get PE attributes
        self.pe=pefile.PE(data=bytez)
        # get file size
        # self.size=os.path.getsize(bytez)
        print("Len:", len(bytez))
        self.size=len(bytez)
        # get file type
        #self.file_type=commands.getstatusoutput("file -b "+bytez)[1]
        # get md5 and sha 1 hashes
        # self.md5=self.extractMD5(bytez)
        # self.sha1=self.extractSHA1(bytez)
        # get fuzzy hash
        #self.fuzzy=commands.getstatusoutput("ssdeep -b " +bytez)
        # get strings
        if strings:
            self.strings=self.extractStrings(bytez)
        else:
            self.ignored_features.append("getStrings")
        # get entropy
        self.entropy=self.extractEntropy(bytez)
        return
    # General Sample Attributes
    # def getName(self):
    # return self.name
    def getSize(self):
        return self.size
    # def getFileType(self):
    # return self.file_type
    # def getMD5(self):
    # return self.md5
    # def getSHA1(self):
    # return self.sha1
    # def getFuzzy(self):
    # return self.fuzzy
    def getStrings(self):
        return self.strings
    def getEntropy(self):
        return self.entropy
    # Dos Header
    def getDOS_HEADER(self):
        return self.pe.DOS_HEADER
    # File Header
    def getFILE_HEADER(self):
        return self.pe.FILE_HEADER
    def getMachine(self):
        return self.pe.FILE_HEADER.Machine
    def getNumberOfSections(self):
        return self.pe.FILE_HEADER.NumberOfSections
    def getTimeDateStamp(self):
        return self.pe.FILE_HEADER.TimeDateStamp
    # def getFormatedTimeDateStamp(self):
    # time = self.pe.FILE_HEADER.TimeDateStamp
    # time = datetime.datetime.fromtimestamp(int(time))
    # return time.strftime('%Y-%m-%d %H:%M:%S')
    def getPointerToSymbolTable(self):
        return self.pe.FILE_HEADER.PointerToSymbolTable
    def getNumberOfSymbols(self):
        return self.pe.FILE_HEADER.NumberOfSymbols
    def getSizeOfOptionalHeader(self):
        return self.pe.FILE_HEADER.SizeOfOptionalHeader
    def getCharacteristics(self):
        return self.pe.FILE_HEADER.Characteristics
    # Optional Header
    def getOPTIONAL_HEADER(self):
        return self.pe.OPTIONAL_HEADER
    def getMagic(self):
        return self.pe.OPTIONAL_HEADER.Magic
    def getSizeOfCode(self):
        return self.pe.OPTIONAL_HEADER.SizeOfCode
    def getSizeOfInitializedData(self):
        return self.pe.OPTIONAL_HEADER.SizeOfInitializedData
    def getSizeOfUninitializedData(self):
        return self.pe.OPTIONAL_HEADER.SizeOfUninitializedData
    # def getEntryPoint(self):
    #     return self.pe.OPTIONAL_HEADER.EntryPoint
    def getBaseOfCode(self):
        return self.pe.OPTIONAL_HEADER.BaseOfCode
    def getBaseOfData(self):
        try:
            return self.pe.OPTIONAL_HEADER.BaseOfData
        except:
            return 0
    def getImageBase(self):
        return self.pe.OPTIONAL_HEADER.ImageBase
    def getSizeOfImage(self):
        return self.pe.OPTIONAL_HEADER.SizeOfImage
    def getSizeOfHeaders(self):
        return self.pe.OPTIONAL_HEADER.SizeOfHeaders
    def getDllCharacteristics(self):
        return self.pe.OPTIONAL_HEADER.DllCharacteristics
    def getFileAlignment(self):
        return self.pe.OPTIONAL_HEADER.FileAlignment
    def getNumberOfRvaAndSizes(self):
        return self.pe.OPTIONAL_HEADER.NumberOfRvaAndSizes
    # PE Type
    def getPE_TYPE(self):
        return self.pe.PE_TYPE
    # Directory Entry Import
    def getDIRECTORY_ENTRY_IMPORT(self):
        try:
            return self.pe.DIRECTORY_ENTRY_IMPORT
        except:
            return []
    def getImportedDlls(self):
        return " ".join([d.dll.decode("utf-8") for d in self.getDIRECTORY_ENTRY_IMPORT()])
    def getImportedSymbols(self):
        symbols = []
        for i in self.getDIRECTORY_ENTRY_IMPORT():
            for s in i.imports:
                if s.name != None:
                    symbols.append(s.name.decode("utf-8"))
        return " ".join(symbols)
    # Directory Entry Export
    # def getDIRECTORY_ENTRY_EXPORT(self):
    #     return self.pe.DIRECTORY_ENTRY_EXPORT
    # Signaturues: packer/compiler/tools/etc
    def getIdentify(self):
        # Load PE Signature Database & Sample PE
        db_path = os.path.dirname(__file__) + "/peid_userdb_uft8.txt"
        sigs=peutils.SignatureDatabase(db_path) #data=open(db_path, 'r').read())
        # Match PE against signature database
        matches=sigs.match_all(self.pe, ep_only=True)
        m = []
        if matches:
            for l in matches:
                for i in l:
                    m.append(str(i))
            return " ".join(m)  
        else:
            return ""
    # return all valid methods for attributes extraction
    def attr_methods(self):
        features=[]
        for i in dir(self):
            method = getattr(self, i)
            if i.startswith('get') and hasattr(method, '__call__') and i not in self.ignored_features:
                features.append(i.replace("get",""))
        return features
    def attr_values(self):
        values = []
        attrs = []
        for i in dir(self):
            method = getattr(self, i)
            if i.startswith('get') and hasattr(method, '__call__') and i not in self.ignored_features:
                attrs.append(i[3:])
                values.append(method())
        return values, attrs
    # util methods
    def extractMD5(self, filename):
        return hashlib.md5(open(filename, 'rb').read()).hexdigest()
    def extractSHA1(self, filename):
        data=open(filename, "rb").read()
        return hashlib.sha1(data).hexdigest()
    def extractStrings(self,filename):
        frag=""
        strList=[]
        bufLen=2048
        FRAG_LEN=4 # Min length to report as string

        fp=open(filename, "rb")

        offset=0
        buf=fp.read(bufLen)
        while buf:
            for char in buf:
                # Uses curses library to locate printable chars
                # in binary files.
                if curses.ascii.isprint(char)==False:
                    if len(frag)>FRAG_LEN:
                        # strList.append([hex(offset-len(frag)),frag])
                        strList.append(frag)
                    frag=""
                else:
                    frag=frag+chr(char)

                offset+=1
            buf=fp.read(bufLen)
        return strList
    def extractEntropy(self, data):
        # data=open(data, "rb")
        # data = data.read()
        if not data:
            return 0
        entropy=0
        for x in range(256):
            p_x = float(data.count(bytes(x)))/len(data)
            if p_x>0:
                entropy += - p_x*math.log(p_x, 2)
        return entropy