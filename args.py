from argparse import ArgumentParser, Namespace
from tool import mkdir

def add_train_argument(p):
    p.add_argument('--input',type=int,default=1024)
    p.add_argument('--hiddenSize',type=int,default=480)
    p.add_argument('--outSize', type=int, default=128)
    p.add_argument('--nodeNum',type=int,default=16)
    p.add_argument('--hdnDropout',type=float,default=0.2)
    p.add_argument('--fcDropout',type=float,default=0.6)
    p.add_argument('--GraphLearner_hidden',type=int,default=60)
    p.add_argument('--update_adj_ratio',type=int,default=0.75)
    p.add_argument('--graph_skip_conn',type=float,default=0.65)
    p.add_argument('--adjDropout',type=int,default=0.5)
    p.add_argument('--eps_adj',type=float,default=4e-05)

    p.add_argument('--epoch',type=int,default=128)
    p.add_argument('--batchSize',type=int,default=512)
    p.add_argument('--trainSize',type=int,default=512)
    p.add_argument('--earlyStop', type=int, default=15)

    p.add_argument('--log_path',type=str,default='log')

    p.add_argument('--save_path',type=str,default='model_save',
                   help='The path to save output model.pt.,default is "model_save/"')


def set_train_argument():
    p = ArgumentParser()
    add_train_argument(p)
    args = p.parse_args()

    mkdir(args.save_path)
    return args

def add_hyper_argument(p):
    p.add_argument('--search_num', type=int,default=100,
                   help='The number of hyperparameters searching.')

def set_hyper_argument():
    p = ArgumentParser()
    add_train_argument(p)
    add_hyper_argument(p)
    args = p.parse_args()

    mkdir(args.save_path)

    args.search_now = 0
    return args