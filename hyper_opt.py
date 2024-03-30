from argparse import ArgumentParser, Namespace
from hyperopt import fmin, tpe, hp
import numpy as np
import os
from copy import deepcopy
from tool import set_log
from args import set_hyper_argument
from use import train
space = {
         'hiddenSize':hp.quniform('hiddenSize', low=128, high=512, q=32),
         'nodeNum':hp.quniform('nodeNum', low=16, high=128, q=16),
         'hdnDropout':hp.quniform('hdnDropout', low=0.0, high=0.8, q=0.1),
         'fcDropout':hp.quniform('fcDropout', low=0.0, high=0.8, q=0.1),
         'GraphLearner_hidden':hp.quniform('GraphLearner_hidden', low=50, high=100, q=10),
         'update_adj_ratio':hp.quniform('update_adj_ratio', low=0.0, high=0.8, q=0.05),
         'graph_skip_conn': hp.quniform('graph_skip_conn', low=0.0, high=0.8, q=0.05),
         'adjDropout': hp.quniform('adjDropout', low=0.0, high=0.8, q=0.1),
}

def fn(space):
    search_no = args.search_now
    log_name = 'train' + str(search_no)
    log = set_log(log_name, args.log_path)
    result_path = os.path.join(args.log_path, 'hyper_para_result.txt')

    # list = ['fp_2_dim', 'nhid', 'nheads']
    # for one in list:
    #     space[one] = int(space[one])

    hyperp = deepcopy(args)
    name_list = []
    for key, value in space.items():
        name_list.append(str(key))
        name_list.append('-')
        name_list.append((str(value))[:5])
        name_list.append('-')
        setattr(hyperp, key, value)
    dir_name = "".join(name_list)
    dir_name = dir_name[:-1]
    hyperp.save_path = os.path.join(hyperp.save_path, dir_name)

    res = train(hyperp, log)
    ave = res['AUC']
    with open(result_path, 'a') as file:
        file.write(str(space) + '\n')
        file.write('Result ' + 'ACC:' + str(res['ACC']) + ' ; ' +'AUC:'+ str(res['AUC']) + ' ;' +'LOSS:'+ str(res['LOSS']) + '\n')

    args.search_now += 1
    return -ave

def hyper_searching(args):
    result_path = os.path.join(args.log_path, 'hyper_para_result.txt')

    result = fmin(fn, space, tpe.suggest, args.search_num)

    with open(result_path, 'a') as file:
        file.write('Best Hyperparameters : \n')
        file.write(str(result) + '\n')

if __name__ == '__main__':
    args = set_hyper_argument()
    hyper_searching(args)