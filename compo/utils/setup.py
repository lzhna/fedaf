import os, time, random, logging, torch
import numpy as np
from compo.models.cnn import CNNCifar
from compo.models.resnet import resnet
from compo.models.vgg import vgg

EXPRS = ['fedavg', 'feddf', 'fedaf', 'dfavg', 'solo']

def workflow(args):
    if args.expr_mode == 'fedavg':
        args.ab_ens_kd = True
        args.ab_cwpa = True
        args.ab_kf = True
    elif args.expr_mode == 'feddf':
        args.ab_cwpa = True
        args.ab_kf = True
    elif args.expr_mode == 'fedaf':
        if not args.ab_repr:
            args.dist_T = None
    
    return args

def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def create_log_dir(expr_mode):
    lt = time.localtime()
    timestamp = time.strftime("%y%m%d[%H%M%S]", lt)

    expr_dir = 'null'
    for algr in EXPRS:
        if algr in expr_mode:
            expr_dir = algr
            break
    else:
        exit(f'{expr_mode} not supported')

    expr_rpath = os.path.join('results', expr_dir)
    if not os.path.exists(expr_rpath):
        os.mkdir(expr_rpath)

    root_path = os.path.join(expr_rpath, timestamp)
    os.mkdir(root_path)
    log_file = os.path.join(root_path, f'expr_{timestamp}.log')
    logging.basicConfig(
        filename=log_file,
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M:%S', level=logging.DEBUG, filemode='a+')
    logging.getLogger('nmslib').setLevel(logging.WARNING)
    
    return root_path, timestamp

class Config:
    def __str__(self):
        return str(self.__dict__)

def load_knn_config(args):
    config = Config()
    config.M = 15
    config.efC = 100
    config.num_threads = 16
    config.K = 1
    config.efS = 100
    config.query_time_params = {'efSearch': config.efS}
    if args.dist_T == None:
        config.index_time_params = {'M': config.M, 'indexThreadQty': config.num_threads, 'efConstruction': config.efC, 'post': 0}
        config.space_name = 'cosinesimil'
        config.method = 'hnsw'
    else:
        config.index_time_params = {'NN': config.M, 'indexThreadQty': config.num_threads, 'efConstruction': config.efC}
        config.space_name = 'kldivgenfast'
        config.method = 'sw-graph'

    return config

def create_model(arch, n_classes):
    if 'resnet' in arch:
        if '-GN' in arch:
            group_norm_num_groups = 2
            resnet_size = int(arch[6:-3])
        else:
            group_norm_num_groups = None
            resnet_size = int(arch[6:])

        group_norm_num_groups = 2 if '-GN' in arch else None
        model = resnet(resnet_size, n_classes, group_norm_num_groups)
    elif 'vgg' in arch:
        if '-GN' in arch:
            use_bn = False
            size = int(arch[3:-3])
        else:
            use_bn = True
            size = int(arch[3:])
        model = vgg(size=size, n_classes=n_classes, vgg_scaling=8, use_bn=use_bn)
    elif 'cnn' in arch:
        model = CNNCifar(num_layers=0, num_classes=n_classes)
    else:
        exit('Error: unrecognized model')

    return model

class MetricsLogger:
    def __init__(self, fpath, title):
        self.fpath = fpath
        self.print_cache = f'{title}\n'
    
    def log(self, line):
        self.print_cache += f'{line}\n'
    
    def print(self):
        with open(self.fpath, 'a+') as f:
            f.write(self.print_cache)
            f.flush()
            self.print_cache = ''

class Trend:
    def __init__(self, patience, higher=True):
        self.patience = patience
        self.higher = higher
        self.best_value = -1e20 if higher else 1e20
        self.best_idnt = 0
        self.count = 0

    def __call__(self, value, idnt):
        cond0 = self.higher and (value > self.best_value)
        cond1 = (not self.higher) and (value < self.best_value)
        if cond0 or cond1:
            self.best_value = value
            self.count = 0
            self.best_idnt = idnt
        else:
            self.count += 1
        
        return self.count >= self.patience
    
    def get_best(self):
        return self.best_value, self.best_idnt

def client_sampling(i_round, n_clients, ratio):
    n_participation = round(n_clients * ratio)
    if n_clients == n_participation:
        client_idxs = [client_idx for client_idx in range(n_clients)]
    else:
        num_clients = min(n_participation, n_clients)
        np.random.seed(i_round)
        client_idxs = np.random.choice(range(n_clients), num_clients, replace=False)
    return client_idxs