from utils import *
from DL_ClassifierModel import *
from args import *
from tool import *


def loaddata():
    dataClass = DataClass(dataPath='data',
                          pSeqMaxLen=1024, dSeqMaxLen=128)

    # load
    with open('data.pkl', 'rb') as file:
        loaded_data = pickle.load(file)
    dataClass.pSeqTokenized = loaded_data['pSeqTokenized']
    dataClass.pSeqTokenized_k = loaded_data['pSeqTokenized_k']
    dataClass.pContFeat = loaded_data['pContFeat']
    dataClass.gSeqTokenized = loaded_data['gSeqTokenized']
    dataClass.pSeqLen = loaded_data['pSeqLen']
    dataClass.dGraphFeat = loaded_data['dGraphFeat']
    dataClass.dFinprFeat = loaded_data['dFinprFeat']
    dataClass.dContFeat = loaded_data['dContFeat']
    dataClass.dAdjMat = loaded_data['dAdjMat']
    dataClass.dSeqTokenized = loaded_data['dSeqTokenized']
    dataClass.dSeqTokenized_k = loaded_data['dSeqTokenized_k']
    dataClass.dSeqLen = loaded_data['dSeqLen']
    dataClass.trainSampleNum =  loaded_data['trainSampleNum']
    dataClass.validSampleNum = loaded_data['validSampleNum']
    dataClass.testSampleNum = loaded_data['testSampleNum']
    dataClass.eSeqData = loaded_data['eSeqData']
    dataClass.edgeLab = loaded_data['edgeLab']
    dataClass.dMolData = loaded_data['dMolData']
    dataClass.smiles = loaded_data['smiles']
    return dataClass


def train(args,log):
    dataClass = loaddata()
    model = AIGO-DTI(args=args,log=log,cSize=dataClass.pContFeat.shape[1])
    res = model.train(dataClass, args=args,stopRounds=-1,log=log,
            savePath='AIGO-DTI', metrics="AUC", report=["ACC", "AUC", "LOSS",'F1','Precision','AUPR'],
            preheat=0)
    return res

if __name__ == '__main__':
    args = set_train_argument()
    log = set_log('train', args.log_path)
    train(args,log)
