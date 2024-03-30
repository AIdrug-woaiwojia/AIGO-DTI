import numpy as np
import pandas as pd
import torch, time, os, pickle, random
from torch import nn as nn
from nnLayer import *
from metrics import *
from collections import Counter, Iterable
from sklearn.model_selection import StratifiedKFold, KFold
from torch.backends import cudnn
from tqdm import tqdm
from Others import *
from graphlearn import *
from gnn import *
from CMPNE import CMPNEncoder, prompt_generator_output, MolGraph, AllMolGraph
from tool import mkdir



class BaseClassifier:
    def __init__(self):
        pass

    def calculate_y_logit(self, X, XLen):
        pass

    def cv_train(self, dataClass, trainSize=256, batchSize=256, epoch=100, stopRounds=10, earlyStop=10, saveRounds=1,
                 optimType='Adam', preheat=5, lr1=0.001, lr2=0.00003, momentum=0.9, weightDecay=0, kFold=5,
                 isHigherBetter=True, metrics="AUC", report=["ACC", "AUC"],
                 savePath='model', seed=9527, loc=-1):
        skf = StratifiedKFold(n_splits=kFold, random_state=seed, shuffle=True)
        validRes = []
        tvIdList = list(
            range(dataClass.trainSampleNum + dataClass.validSampleNum))  # dataClass.trainIdList+dataClass.validIdList
        # self._save_emb('cache/_preEmbedding.pkl')
        for i, (trainIndices, validIndices) in enumerate(skf.split(tvIdList, [i[2] for i in dataClass.eSeqData])):
            print(f'CV_{i + 1}:')
            if loc > 0 and i + 1 != loc:
                print(f'Pass CV_{i + 1}')
                continue
            self.reset_parameters()
            # self._load_emb('cache/_preEmbedding.pkl')
            dataClass.trainIdList, dataClass.validIdList = trainIndices, validIndices
            dataClass.trainSampleNum, dataClass.validSampleNum = len(trainIndices), len(validIndices)
            res = self.train(dataClass, trainSize, batchSize, epoch, stopRounds, earlyStop, saveRounds, optimType,
                             preheat, lr1, lr2, momentum, weightDecay,
                             isHigherBetter, metrics, report, f"{savePath}_cv{i + 1}")
            validRes.append(res)
        Metrictor.table_show(validRes, report)

    def cv_train_by_protein(self, dataClass, trainSize=256, batchSize=256, epoch=100, stopRounds=10, earlyStop=10,
                            saveRounds=1,
                            optimType='Adam', preheat=5, lr1=0.001, lr2=0.00003, momentum=0.9, weightDecay=0, kFold=5,
                            isHigherBetter=True, metrics="AUC", report=["ACC", "AUC"],
                            savePath='model', seed=9527, loc=-1):
        kf = KFold(n_splits=kFold, random_state=seed, shuffle=True)
        validRes = []
        proteins = list(range(len(dataClass.p2id)))  # dataClass.trainIdList+dataClass.validIdList
        # self._save_emb('cache/_preEmbedding.pkl')
        for i, (trainProteins, validProteins) in enumerate(kf.split(proteins)):
            print(f'CV_{i + 1}:')
            if loc > 0 and i + 1 != loc:
                print(f'Pass CV_{i + 1}')
                continue
            self.reset_parameters()
            # self._load_emb('cache/_preEmbedding.pkl')

            dataClass.trainIdList = [i for i in range(len(dataClass.eSeqData)) if
                                     dataClass.eSeqData[i, 0] in trainProteins]
            dataClass.validIdList = [i for i in range(len(dataClass.eSeqData)) if
                                     dataClass.eSeqData[i, 0] in validProteins]
            dataClass.trainSampleNum, dataClass.validSampleNum = len(dataClass.trainIdList), len(dataClass.validIdList)

            res = self.train(dataClass, trainSize, batchSize, epoch, stopRounds, earlyStop, saveRounds, optimType,
                             preheat, lr1, lr2, momentum, weightDecay,
                             isHigherBetter, metrics, report, f"{savePath}_cv{i + 1}")
            validRes.append(res)
        Metrictor.table_show(validRes, report)

    def get_optimizer(self, optimType, lr, weightDecay, momentum):
        if optimType == 'Adam':
            return torch.optim.Adam(self.moduleList.parameters(), lr=lr, weight_decay=weightDecay)
        elif optimType == 'AdamW':
            return torch.optim.AdamW(self.moduleList.parameters(), lr=lr, weight_decay=weightDecay)
        elif optimType == 'SGD':
            return torch.optim.SGD(self.moduleList.parameters(), lr=lr, momentum=momentum, weight_decay=weightDecay)

    def train(self, dataClass, args, log, stopRounds=10, saveRounds=1,
              optimType='Adam', preheat=5, lr1=0.001, lr2=0.00003, momentum=0.9, weightDecay=0, isHigherBetter=True,
              metrics="AUC", report=["ACC", "AUC"],
              savePath='model'):
        info = log.info
        debug = log.debug

        mkdir(args.save_path)
        trainSize = args.trainSize
        batchSize = args.batchSize
        epoch = args.epoch
        earlyStop = args.earlyStop


        dataClass.describe()
        assert batchSize % trainSize == 0
        metrictor = Metrictor(info)
        self.stepCounter = 0
        self.stepUpdate = batchSize // trainSize
        self.preheat()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.moduleList.parameters()), lr=lr1,
                                     weight_decay=weightDecay)
        schedulerRLR = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max' if isHigherBetter else 'min',
                                                                  factor=0.5, patience=4,
                                                                  verbose=True)
        trainStream = dataClass.random_batch_data_stream(batchSize=trainSize, type='train', sampleType=self.sampleType,
                                                         device=self.device)
        itersPerEpoch = (dataClass.trainSampleNum + trainSize - 1) // trainSize
        mtc, bestMtc, stopSteps = 0.0, 0.0, 0
        if dataClass.validSampleNum > 0: validStream = dataClass.random_batch_data_stream(batchSize=trainSize,
                                                                                          type='valid',
                                                                                          sampleType=self.sampleType,
                                                                                          device=self.device, log=True)
        st = time.time()
        debug('Training Model')

        for e in range(epoch):
            if e == preheat:
                if preheat > 0:
                    self.load(args.save_path + '.pkl')
                self.normal()
                optimizer = self.get_optimizer(optimType=optimType, lr=lr2, weightDecay=weightDecay, momentum=momentum)
                # self.schedulerWU = ScheduledOptim(optimizer, lr2, 1000)
                schedulerRLR = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                          mode='max' if isHigherBetter else 'min',
                                                                          factor=0.5, patience=30, verbose=True)
            for i in tqdm(range(itersPerEpoch)):

                self.to_train_mode()
                X, Y = next(trainStream)
                if X['res']:
                    loss = self._train_step(X, Y, optimizer, args)
                if stopRounds > 0 and (e * itersPerEpoch + i + 1) % stopRounds == 0:
                    self.to_eval_mode()
                    info(f"After iters {e * itersPerEpoch + i + 1}: [train] loss= {loss:.3f};")
                    if dataClass.validSampleNum > 0:
                        X, Y = next(validStream)
                        loss = self.calculate_loss(X, Y, args)
                        info(f' [valid] loss= {loss:.3f}')
                    restNum = ((itersPerEpoch - i - 1) + (epoch - e - 1) * itersPerEpoch) * trainSize
                    speed = (e * itersPerEpoch + i + 1) * trainSize / (time.time() - st)
                    info(" speed: %.3lf items/s; remaining time: %.3lfs;" % (speed, restNum / speed))

            if dataClass.validSampleNum > 0 and (e + 1) % saveRounds == 0:
                self.to_eval_mode()
                info(f'========== Epoch:{e + 1:5d} ==========')
                Y_pre, Y = self.calculate_y_prob_by_iterator(
                    dataClass.one_epoch_batch_data_stream(trainSize, type='train', mode='predict', device=self.device),args)
                metrictor.set_data(Y_pre, Y)
                info(f'[Total Train]')
                metrictor(report)
                info(f'[Total Valid]' )
                Y_pre, Y = self.calculate_y_prob_by_iterator(
                    dataClass.one_epoch_batch_data_stream(trainSize, type='valid', mode='predict', device=self.device),args)
                metrictor.set_data(Y_pre, Y)
                res = metrictor(report)
                mtc = res[metrics]
                schedulerRLR.step(mtc)
                info('=================================')
                if (mtc > bestMtc and isHigherBetter) or (mtc < bestMtc and not isHigherBetter):
                    info(f'Bingo!!! Get a better Model with val {metrics}: {mtc:.3f}!!!')
                    bestMtc = mtc
                    self.save(os.path.join(args.save_path,'model.pkl'), e + 1, bestMtc, dataClass)

                    stopSteps = 0
                else:
                    stopSteps += 1
                    if stopSteps >= earlyStop:

                        info(f'The val {metrics} has not improved for more than {earlyStop} steps in epoch {e + 1}, stop training.')

                        break

        # savePath = 'model.pkl'
        # self.load(savePath)
        self.load(os.path.join(args.save_path,'model.pkl'))
        self.to_eval_mode()

        bestMtc_decimal = "%.3lf" % bestMtc
        new_filename = f"model_{bestMtc_decimal}.pkl"
        old_path = os.path.join(args.save_path, 'model.pkl')
        new_path = os.path.join(args.save_path, new_filename)
        os.rename(old_path, new_path)

        info(f'============ Result ============')

        info(f'[Total Train]')

        Y_pre, Y = self.calculate_y_prob_by_iterator(
            dataClass.one_epoch_batch_data_stream(trainSize, type='train', mode='predict', device=self.device),args=args)
        metrictor.set_data(Y_pre, Y)
        metrictor(report)
        info(f'[Total Valid]')

        Y_pre, Y = self.calculate_y_prob_by_iterator(
            dataClass.one_epoch_batch_data_stream(trainSize, type='valid', mode='predict', device=self.device),args=args)
        metrictor.set_data(Y_pre, Y)
        res = metrictor(report)
        if dataClass.testSampleNum > 0:
            info(f'[Total Test]')

            Y_pre, Y = self.calculate_y_prob_by_iterator(
                dataClass.one_epoch_batch_data_stream(trainSize, type='test', mode='predict_all', device=self.device),args=args)
            metrictor.set_data(Y_pre, Y)
            metrictor(report)
        # metrictor.each_class_indictor_show(dataClass.id2lab)
        res = metrictor(report)
        info(f'================================')

        return res

    def reset_parameters(self):
        for module in self.moduleList:
            for subModule in module.modules():
                if hasattr(subModule, "reset_parameters"):
                    subModule.reset_parameters()

    def save(self, path, epochs, bestMtc=None, dataClass=None):
        stateDict = {'epochs': epochs, 'bestMtc': bestMtc}
        for module in self.moduleList:
            stateDict[module.name] = module.state_dict()
        if dataClass is not None:
            # stateDict['trainIdList'],stateDict['validIdList'],stateDict['testIdList'] = dataClass.trainIdList,dataClass.validIdList,dataClass.testIdList
            if 'am2id' in stateDict:
                stateDict['am2id'], stateDict['id2am'] = dataClass.am2id, dataClass.id2am
            if 'go2id' in stateDict:
                stateDict['go2id'], stateDict['id2go'] = dataClass.go2id, dataClass.id2go
            if 'at2id' in stateDict:
                stateDict['at2id'], stateDict['id2at'] = dataClass.at2id, dataClass.id2at
        torch.save(stateDict, path)
        print('Model saved in "%s".' % path)

    def load(self, path, map_location=None, dataClass=None):
        parameters = torch.load(path, map_location=map_location)
        for module in self.moduleList:
            module.load_state_dict(parameters[module.name])
        if dataClass is not None:
            # if "trainIdList" in parameters:
            #     dataClass.trainIdList = parameters['trainIdList']
            # if "validIdList" in parameters:
            #     dataClass.validIdList = parameters['validIdList']
            # if "testIdList" in parameters:
            #     dataClass.testIdList = parameters['testIdList']
            if 'am2id' in parameters:
                dataClass.am2id, dataClass.id2am = parameters['am2id'], parameters['id2am']
            if 'go2id' in parameters:
                dataClass.go2id, dataClass.id2go = parameters['go2id'], parameters['id2go']
            if 'at2id' in parameters:
                dataClass.at2id, dataClass.id2at = parameters['at2id'], parameters['id2at']
        print("%d epochs and %.3lf val Score 's model load finished." % (parameters['epochs'], parameters['bestMtc']))

    def _save_emb(self, path):
        stateDict = {}
        for module in self.embModuleList:
            stateDict[module.name] = module.state_dict()
        torch.save(stateDict, path)
        print('Pre-trained Embedding saved in "%s".' % path)

    def _load_emb(self, path, map_location=None):
        parameters = torch.load(path, map_location=map_location)
        for module in self.embModuleList:
            module.load_state_dict(parameters[module.name])
        print('Pre-trained Embedding loaded in "%s".' % path)

    def preheat(self):
        for param in self.finetunedEmbList.parameters():
            param.requires_grad = False

    def normal(self):
        for param in self.finetunedEmbList.parameters():
            param.requires_grad = True

    def calculate_y_prob(self, X, Y, mode,args):
        Y_pre = self.calculate_y_logit(X, Y,args = args, mode = mode)['y_logit']
        return torch.sigmoid(Y_pre)

    # def calculate_y(self, X):
    #     Y_pre = self.calculate_y_prob(X)
    #     return torch.argmax(Y_pre, dim=1)
    def calculate_loss(self, X, Y, args):
        loss = self.calculate_y_logit(X, Y, args = args, mode = 'train')

        # out = self.calculate_y_logit(X,Y, 'predict')
        # Y_logit = out['y_logit']
        #
        # addLoss = 0.0
        # if 'loss' in out: addLoss += out['loss']
        # return self.criterion(Y_logit, Y) + addLoss

        return loss

    def calculate_indicator_by_iterator(self, dataStream, classNum, report):
        metrictor = Metrictor(classNum)
        Y_prob_pre, Y = self.calculate_y_prob_by_iterator(dataStream)
        metrictor.set_data(Y_prob_pre, Y)
        return metrictor(report)

    def calculate_y_prob_by_iterator(self, dataStream,args):
        YArr, Y_preArr = [], []
        while True:
            try:
                X, Y = next(dataStream)
            except:
                break
            Y_pre, Y = self.calculate_y_prob(X, Y=Y, mode='predict',args=args).cpu().data.numpy(), Y.cpu().data.numpy()
            YArr.append(Y)
            Y_preArr.append(Y_pre)
        YArr, Y_preArr = np.hstack(YArr).astype('int32'), np.hstack(Y_preArr).astype('float32')
        return Y_preArr, YArr

    # def calculate_y_by_iterator(self, dataStream):
    #     Y_preArr, YArr = self.calculate_y_prob_by_iterator(dataStream)
    #     return Y_preArr.argmax(axis=1), YArr
    def to_train_mode(self):
        for module in self.moduleList:
            module.train()

    def to_eval_mode(self):
        for module in self.moduleList:
            module.eval()

    def _train_step(self, X, Y, optimizer, args):
        self.stepCounter += 1
        if self.stepCounter < self.stepUpdate:
            p = False
        else:
            self.stepCounter = 0
            p = True
        loss = self.calculate_loss(X, Y, args) / self.stepUpdate
        loss.backward()

        if p:
            nn.utils.clip_grad_norm_(self.moduleList.parameters(), max_norm=20, norm_type=2)
            optimizer.step()
            optimizer.zero_grad()
            # self.schedulerWU.step_and_update_lr()
            # self.schedulerWU.zero_grad()
        return loss * self.stepUpdate

class AIGO_DTI(BaseClassifier):
    def __init__(self, args, log, cSize, resnet=True, sampleType='CEL',
                 useFeatures={"kmers": True, "pSeq": True, "FP": True, "dSeq": True},
                 maskDTI=False):
        info = log.info
        debug = log.debug

        debug(f'Running building model.')

        outSize = args.outSize
        cHiddenSizeList = [args.input]
        fHiddenSizeList = [args.input, int(args.hiddenSize)]
        gcnHiddenSizeList = [outSize, outSize]
        fcHiddenSizeList = [outSize]
        nodeNum = int(args.nodeNum)
        hdnDropout = args.hdnDropout
        fcDropout = args.fcDropout
        device = torch.device('cuda')

        self.nodeEmbedding = TextEmbedding(
            torch.tensor(np.random.normal(size=(max(nodeNum, 0), outSize)), dtype=torch.float32), dropout=hdnDropout,
            name='nodeEmbedding').to(device)

        self.amEmbedding = TextEmbedding(torch.eye(24), dropout=hdnDropout, freeze=True, name='amEmbedding').to(device)
        self.pCNN = TextCNN(24, 64, [25], ln=True, name='pCNN').to(device)
        self.pFcLinear = MLP(64, outSize, dropout=hdnDropout, bnEveryLayer=True, dpEveryLayer=True, outBn=True,
                             outAct=True, outDp=True, name='pFcLinear').to(device)

        # self.dCNN = TextCNN(75, 64, [7], ln=True, name='dCNN').to(device)
        # self.dFcLinear = MLP(64, outSize, dropout=hdnDropout, bnEveryLayer=True, dpEveryLayer=True, outBn=True,
        #                      outAct=True, outDp=True, name='dFcLinear').to(device)

        self.fgLinear = MLP(1024, outSize, fHiddenSizeList, outAct=True, name='fgLinear', dropout=hdnDropout, dpEveryLayer=True, outDp=True, bnEveryLayer=True, outBn=True).to(device)

        # self.fFcLinear = MLP(fSize, outSize, fHiddenSizeList, outAct=True, name='fFcLinear', dropout=hdnDropout, dpEveryLayer=True, outDp=True, bnEveryLayer=True, outBn=True).to(device)
        self.fFcLinear = MLP(300, 128, fHiddenSizeList, outAct=True, name='fFcLinear', dropout=hdnDropout,
                             dpEveryLayer=True, outDp=True, bnEveryLayer=True, outBn=True).to(device)

        self.cFcLinear = MLP(cSize, outSize, cHiddenSizeList, outAct=True, name='cFcLinear', dropout=hdnDropout,
                             dpEveryLayer=True, outDp=True, bnEveryLayer=True, outBn=True).to(device)

        self.nodeGCN = GCN(outSize, outSize, gcnHiddenSizeList, name='nodeGCN', dropout=hdnDropout, dpEveryLayer=True,
                           outDp=True, bnEveryLayer=True, outBn=True, resnet=resnet).to(device)

        self.fcLinear = MLP(outSize, 1, fcHiddenSizeList, dropout=fcDropout, bnEveryLayer=True, dpEveryLayer=True).to(
            device)
        self.graph_learner = GraphLearner(input_size=outSize, hidden_size=args.GraphLearner_hidden,
                                          ##config['graph_learn_hidden_size']=70
                                          topk=None,
                                          epsilon=0,
                                          num_pers=4,
                                          metric_type="weighted_cosine",
                                          name='graph_learner').to(device)

        self.graph_learner2 = GraphLearner(input_size=outSize, hidden_size=args.GraphLearner_hidden,
                                           ##config['graph_learn_hidden_size']=70
                                           topk=None,
                                           epsilon=0,
                                           num_pers=4,
                                           metric_type="weighted_cosine",
                                           name="graph_learner2").to(device)
        self.CMPN = CMPNEncoder(atom_fdim=133, bond_fdim=147)
        self.CMPN.W_i_atom = prompt_generator_output()(self.CMPN.W_i_atom)
        self.CMPN = self.CMPN.to(device)
        # self.GCN = GCN(nfeat=128,nhid=)

        self.criterion = nn.BCEWithLogitsLoss()

        self.embModuleList = nn.ModuleList([])
        self.finetunedEmbList = nn.ModuleList([])
        self.moduleList = nn.ModuleList(
            [self.nodeEmbedding, self.cFcLinear, self.fFcLinear, self.nodeGCN, self.fcLinear,self.fgLinear,
             self.amEmbedding, self.pCNN, self.pFcLinear, self.graph_learner,
             self.graph_learner2, self.CMPN])
        self.sampleType = sampleType
        self.device = device
        self.resnet = resnet
        self.nodeNum = nodeNum
        self.hdnDropout = hdnDropout
        self.useFeatures = useFeatures
        self.maskDTI = maskDTI

        # debug(self.moduleList)

    def Mol2Feature(self, smiles):
        all = self.Mol2graph(smiles)
        molfeature = self.CMPN.forward(all)
        return molfeature

    def Mol2graph(self, smiles):
        mol_graphs = []
        for smile in smiles:
            mol_graph = MolGraph(smile)
            mol_graphs.append(mol_graph)
        return AllMolGraph(mol_graphs)

    def calculate_y_logit(self, X, Y,args,
                          mode='train'):
        self.graph_learn_regularization = True
        mol_feature = self.Mol2Feature(smiles=X['smiles'])

        Xam = (self.cFcLinear(X['aminoCtr']).unsqueeze(1) if self.useFeatures['kmers'] else 0) + \
              (self.pFcLinear(self.pCNN(self.amEmbedding(X['aminoSeq']))).unsqueeze(1) if self.useFeatures[
                  'pSeq'] else 0)
        Xat = (self.fFcLinear(mol_feature).unsqueeze(1) if self.useFeatures['FP'] else 0) + \
              (self.fgLinear(X['atomFin']).unsqueeze(1) if self.useFeatures['FP'] else 0)

        if self.nodeNum > 0:
            node = self.nodeEmbedding.dropout2(self.nodeEmbedding.dropout1(self.nodeEmbedding.embedding.weight)).repeat(
                len(Xat), 1, 1)
            node = torch.cat([Xam, Xat, node], dim=1)
            nodeDist = torch.sqrt(torch.sum(node ** 2, dim=2,
                                            keepdim=True) + 1e-8)
            cosNode = torch.matmul(node, node.transpose(1, 2)) / (nodeDist * nodeDist.transpose(1,
                                                                                                2) + 1e-8)

            cosNode[cosNode < 0] = 0
            cosNode[:, range(node.shape[1]),
            range(node.shape[1])] = 1

            cur_raw_adj, cur_adj = learn_graph(self=self, graph_learner=self.graph_learner, node_features=node,
                                               graph_skip_conn=args.graph_skip_conn, graph_include_self=False, init_adj=cosNode)
            cur_raw_adj = F.dropout(cur_raw_adj, args.adjDropout, training=False)
            cur_adj = F.dropout(cur_adj, args.adjDropout, training=False)

            node_vec = torch.relu(self.nodeGCN(node, cur_adj))
            node_vec = F.dropout(node_vec, args.adjDropout, training=False)

            node_embed = node_vec[:, 0, :] * node_vec[:, 1,
                                             :]
            output = {"y_logit": self.fcLinear(node_embed).squeeze(dim=1)}  # , "loss":1*l2}
            Y_logit = output['y_logit']
            addLoss = 0.0
            if 'loss' in output: addLoss += output['loss']
            loss1 = self.criterion(Y_logit, Y) + addLoss



            first_raw_adj, first_adj = cur_raw_adj, cur_adj

            max_iter_ = 10
            pre_raw_adj = cur_raw_adj
            pre_adj = cur_adj

            loss = 0
            iter_ = 0
            while (iter_ == 0 or diff(cur_raw_adj, pre_raw_adj,
                                      first_raw_adj).item() > args.eps_adj) and iter_ < max_iter_:  ##diff(cur_raw_adj, pre_raw_adj, first_raw_adj).item() > eps_adj表示这一次迭代得到的adj与上次adj之间的差值，与阈值之间进行比较，若小于阈值，则不断迭代
                iter_ += 1
                pre_adj = cur_adj
                pre_raw_adj = cur_raw_adj
                cur_raw_adj, cur_adj = learn_graph(self=self, graph_learner=self.graph_learner2, node_features=node_vec,
                                                   graph_skip_conn=args.graph_skip_conn, graph_include_self=False,
                                                   init_adj=cosNode)

                if args.update_adj_ratio is not None:
                    cur_adj = args.update_adj_ratio * cur_adj + (1 - args.update_adj_ratio) * first_adj

                node_vec = torch.relu(self.nodeGCN(node, cur_adj))
                node_vec = F.dropout(node_vec, args.adjDropout, training=False)
                node_embed = node_vec[:, 0, :] * node_vec[:, 1,
                                                 :]

                output = {"y_logit": self.fcLinear(node_embed).squeeze(dim=1)}  # , "loss":1*l2}
                Y_logit = output['y_logit']
                if 'loss' in output: addLoss += output['loss']
                loss += self.criterion(Y_logit, Y) + addLoss


            if iter_ > 0:
                loss = loss / iter_ + loss1
            else:
                loss = loss1

        if mode == 'train':
            return loss
        else:
            return output


def diff(X, Y, Z):
    assert X.shape == Y.shape
    diff_ = torch.sum(torch.pow(X - Y, 2))
    norm_ = torch.sum(torch.pow(Z, 2))
    diff_ = diff_ / torch.clamp(norm_, min=1e-12)
    return diff_


def get_index(seqData, sP, sD):
    sPsD = [i[0] in sP and i[1] in sD for i in seqData]
    sPuD = [i[0] in sP and i[1] not in sD for i in seqData]
    uPsD = [i[0] not in sP and i[1] in sD for i in seqData]
    uPuD = [i[0] not in sP and i[1] not in sD for i in seqData]
    return sPsD, sPuD, uPsD, uPuD


def learn_graph(self, graph_learner, node_features, graph_skip_conn=None, node_mask=None, anchor_mask=None,
                graph_include_self=False, init_adj=None, anchor_features=None):
    VERY_SMALL_NUMBER = 1e-12
    self.graph_learn = True
    self.scalable_run = False

    if self.graph_learn:
        if self.scalable_run:
            node_anchor_adj = graph_learner(node_features, anchor_features, node_mask, anchor_mask)
            return node_anchor_adj

        else:
            raw_adj = graph_learner(node_features, node_mask)

            adj = raw_adj / torch.clamp(torch.sum(raw_adj, dim=-1, keepdim=True), min=VERY_SMALL_NUMBER)

            if graph_skip_conn in (0, None):
                if graph_include_self:
                    adj = adj + to_cuda(torch.eye(adj.size(0)), self.device)
            else:
                adj = graph_skip_conn * init_adj + (1 - graph_skip_conn) * adj

            return raw_adj, adj

    else:
        raw_adj = None
        adj = init_adj

        return raw_adj, adj


def to_cuda(x, device=None):
    if device:
        x = x.to(device)
    return x




