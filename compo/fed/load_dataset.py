import numpy as np
from torchvision import transforms
from compo.utils.datasets import *
from compo.utils.setup import init_seed
import pickle, os, logging, torch


NORM = {
    'cifar10': [[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]],
    'cifar100': [[0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]],
    'cinic10': [[0.47889522, 0.47227842, 0.43047404], [0.24205776, 0.23828046, 0.25874835]],
    'tiny': [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
}

NC = {
    'cifar10': 10,
    'cifar100': 100,
    'cinic10': 10,
    'tiny': 200
}

def get_transform(dname, use_norm=False, dsize=None):
    trans = [transforms.ToTensor()]
    if use_norm:
        trans.append(transforms.Normalize(*NORM[dname]))
    
    if dsize is not None:
        trans.append(transforms.RandomCrop(dsize, padding=4))

    return transforms.Compose(trans)

def get_dataset(name, data_dir='data', is_train=True, use_norm=False, dsize=None, labeled=True):
    if 'RN-' in name:
        fpath = os.path.join(data_dir, 'random_noise', f'{name}.pt')
        data = torch.load(fpath, map_location='cpu')
        ds = BasicDataset(data, None, None, None)
        return ds

    transform = get_transform(name, use_norm, dsize)
    if 'cifar' in name:
        ds = CIFAR(name, data_dir, is_train, transform=transform, target_transform=None, download=True, labeled=labeled)
    elif 'cinic' in name:
        ds = Imgs('cinic10', data_dir, is_train, transform=transform, target_transform=None, labeled=labeled)
    elif 'tiny' in name:
        ds = ImageFolder_custom(root=os.path.join(data_dir, 'tiny-imagenet-200', 'train'), train=True)
    else:
        exit(f'dataset {name} is not supported')
    
    return ds

def usample_idxs(label_dict:dict, num, data_seed):
    init_seed(data_seed)
    n_per_label = num // len(label_dict)
    n_remain = num % len(label_dict)
    labels = list(label_dict.keys())
    elabels = list(np.random.choice(labels, n_remain, replace=False))
    sampled_idxs = []
    slabel_dict = {}
    for c in labels:
        num = n_per_label + 1 if c in elabels else n_per_label
        s = np.random.choice(label_dict[c], num, replace=False)
        sampled_idxs.extend(s)
        slabel_dict[c] = list(s)
        label_dict[c] = list(set(label_dict[c]) - set(s))
    
    return sampled_idxs, slabel_dict

def idx2label(label_dict:dict, idxs):
    lcount = {}
    for idx in idxs:
        for c in label_dict.keys():
            if idx in label_dict[c]:
                lcount.setdefault(c, 0)
                lcount[c] += 1
                break
    return lcount

def record_label_distr(label_dict:dict, client_dataidx_map:dict, distr_fpath):
    labels = list(label_dict.keys())
    labels.sort()
    client_list = list(client_dataidx_map.keys())
    client_list.sort()

    label_distr = {i_client:idx2label(label_dict, idxs)
                    for i_client, idxs in client_dataidx_map.items()}

    lines = 'i_client,' + ','.join([str(e) for e in labels]) + ',sum\n'
    n_total = 0
    sum_per_label = {c:0 for c in labels}
    for i_client in client_list:
        cur = [str(i_client)]
        for c in labels:
            if c in label_distr[i_client]:
                sum_per_label[c] += label_distr[i_client][c]
                cur.append(str(label_distr[i_client][c]))
            else:
                cur.append('0')
        
        npc = len(client_dataidx_map[i_client])
        cur.append(str(npc))
        n_total += npc
        lines += ','.join(cur) + '\n'
    
    cur = ['sum']
    cur.extend([str(sum_per_label[c]) for c in labels])
    cur.append(str(n_total))
    lines += ','.join(cur) + '\n'

    with open(distr_fpath, 'w+') as f_info:
        f_info.write(lines)
        f_info.flush()

def partition_data(label_dict:dict, args, gd_dir):
    pname = f'{args.priv_ds}-A[{args.d_alpha}]-NC[{args.n_clients}]-S[{args.data_seed}-{args.seed}]-P[{args.r_proxy}]-V[{args.r_gval}]'
    idx_map_fpath = os.path.join(gd_dir, f'm-{pname}.pkl')
    distr_fpath = os.path.join(gd_dir, f'd-{pname}.csv')

    if os.path.exists(idx_map_fpath):
        with open(idx_map_fpath, 'rb') as f:
            client_dataidx_map = pickle.load(f)
        return client_dataidx_map

    min_size = 0
    min_require_size = 10
    N = sum([len(label_dict[e]) for e in label_dict.keys()])
    client_dataidx_map = {}

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(args.n_clients)]
        # for each class in the dataset
        for c in label_dict.keys():
            idxs_c = label_dict[c]
            np.random.shuffle(idxs_c)
            proportions = np.random.dirichlet(np.repeat(args.d_alpha, args.n_clients))
            ## Balance
            proportions = np.array([p * (len(idx_j) < N / args.n_clients) for p, idx_j in zip(proportions, idx_batch)])

            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idxs_c)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idxs_c, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(args.n_clients):
        np.random.shuffle(idx_batch[j])
        client_dataidx_map[j] = idx_batch[j]

    with open(idx_map_fpath, 'wb') as f:
        pickle.dump(client_dataidx_map, f)

    record_label_distr(label_dict, client_dataidx_map, distr_fpath)
    return client_dataidx_map

def minus_idxs(label_dict:dict, idxs):
    slabel_dict = {}
    for c in label_dict.keys():
        slabel_dict[c] = list(set(label_dict[c]) & set(idxs))
        label_dict[c] = list(set(label_dict[c]) - set(idxs))
        
    return slabel_dict

def split_gidxs(fpath, label_dict, num, data_seed):
    if os.path.exists(fpath):
        with open(fpath, 'rb') as f:
            idxs = pickle.load(f)
        slabel_dict = minus_idxs(label_dict, idxs)
    else:
        idxs, slabel_dict = usample_idxs(label_dict, num, data_seed)
        with open(fpath, 'wb') as f:
            pickle.dump(idxs, f)
    
    return idxs, slabel_dict

def load_fed_data(args, gd_dir='pdmap'):
    priv_ds = get_dataset(name=args.priv_ds, 
                        use_norm=args.use_norm, 
                        dsize=args.dsize)
    label_dict = priv_ds.index_labels()

    # fpath_nothing = os.path.join(gd_dir, f'{args.priv_ds}-S[{args.data_seed}]-nothing-P[{args.r_proxy}].pkl')
    # idx_nothing, slabel_dict = split_gidxs(fpath=fpath_nothing,
    #                         label_dict=label_dict,
    #                         num=45000,
    #                         data_seed=args.data_seed)

    n_raw = sum([len(label_dict[e]) for e in label_dict.keys()])

    fpath_proxy = os.path.join(gd_dir, f'{args.priv_ds}-S[{args.data_seed}]-proxy-P[{args.r_proxy}].pkl')
    idxs_proxy, slabel_dict = split_gidxs(fpath=fpath_proxy,
                            label_dict=label_dict,
                            # num=int(5000 * args.r_proxy),
                            num=int(n_raw * args.r_proxy),
                            data_seed=args.data_seed)

    if args.sub_proxy is not None:
        fpath_subp = os.path.join(gd_dir, f'{args.priv_ds}-S[{args.data_seed}]-subp[{args.sub_proxy}]-P[{args.r_proxy}].pkl')
        idxs_proxy, _ = split_gidxs(fpath=fpath_subp,
                                label_dict=slabel_dict,
                                num=int(len(idxs_proxy) * args.sub_proxy),
                                data_seed=args.data_seed)

    fpath_gval = os.path.join(gd_dir, f'{args.priv_ds}-S[{args.data_seed}]-gval-P[{args.r_proxy}]-V[{args.r_gval}].pkl')
    idxs_gval, _ = split_gidxs(fpath=fpath_gval,
                            label_dict=label_dict,
                            # num=int(5000 * args.r_gval),
                            num=int(n_raw * args.r_gval),
                            data_seed=args.data_seed)
    
    logging.debug(f'idxs_proxy & idxs_gval = {len(set(idxs_proxy)&set(idxs_gval))}')
    
    client_dataidx_map = partition_data(label_dict, args, gd_dir)
    label_distr = {i_client:idx2label(label_dict, idxs)
                    for i_client, idxs in client_dataidx_map.items()}

    if args.priv_ds == args.publ_ds:
        proxy_set = trunc_ds(priv_ds, idxs_proxy, labeled=False)
    elif args.publ_ds == 'tiny':
        proxy_set = get_dataset(name=args.publ_ds,dsize=32)
        logging.debug(f'tiny = {len(proxy_set)}')
        pdict = proxy_set.index_labels()
        logging.debug(f'pdict = {len(pdict)}')
        fpath = os.path.join(gd_dir, f'tiny-{args.np}-S[{args.data_seed}].pkl')
        idxs_proxy, _ = split_gidxs(fpath, pdict, args.np, args.data_seed)
        transform = get_transform(args.publ_ds, args.use_norm, 32)
        proxy_set = ImageFolder_custom(root=os.path.join('data', 'tiny-imagenet-200', 'train'),
                                    dataidxs=idxs_proxy,
                                    train=True,
                                    transform=transform,
                                    labeled=False)
    else:
        proxy_set = get_dataset(name=args.publ_ds, 
                        use_norm=args.use_norm, 
                        dsize=args.dsize,
                        labeled=False)

    gval_set = trunc_ds(priv_ds, idxs_gval, labeled=True)
    cpriv_sets = {i:trunc_ds(priv_ds, client_dataidx_map[i], labeled=True)
                    for i in client_dataidx_map.keys()}
    gtest_set = get_dataset(name=args.priv_ds,
                        is_train=False,
                        use_norm=args.use_norm,
                        dsize=args.dsize)
    
    logging.debug(f'n_proxy = {len(proxy_set)}, n_gval = {len(gval_set)}, n_gtest = {len(gtest_set)}')
    total_priv = 0
    for i in cpriv_sets.keys():
        n_priv = len(cpriv_sets[i])
        logging.debug(f'client[{i:03d}]: n_priv = {n_priv}')
        total_priv += n_priv
    logging.debug(f'total_priv = {total_priv}')
    
    return proxy_set, gval_set, cpriv_sets, gtest_set, label_distr
