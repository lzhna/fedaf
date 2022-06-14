from torch.utils.data import DataLoader, Dataset
from compo.utils.setup import Config
from compo.fed.model_trainer import ModelTrainer
import nmslib

class Client:
    def __init__(self, args, priv_ds:Dataset, device_idx:int, aux:Config=None):
        self.args = args
        self.device_idx = device_idx
        self.aux = aux

        self.tloader = DataLoader(priv_ds, batch_size=args.ltrain_bs, shuffle=True)
        self.tloader_fixed = DataLoader(priv_ds, batch_size=args.static_bs, shuffle=False)
        self.n_train = len(priv_ds)

        self.assessed_mlist = []
    
    def estim(self, trainer:ModelTrainer, model, pc_bases, pc_idxs, m_info):
        lc_bases, _ = trainer.get_cls_bases(model=model,
                                            data_loader=self.tloader_fixed,
                                            correct_only=True,
                                            label_list=self.aux.local_label,
                                            T=self.args.dist_T)
        
        dists = {}
        config = self.aux.config
        for pseudo_c in pc_bases.keys():
            if pseudo_c in lc_bases:
                index = nmslib.init(method=config.method,
                                    space=config.space_name,
                                    data_type=nmslib.DataType.DENSE_VECTOR)
                index.addDataPointBatch(lc_bases[pseudo_c])
                index.createIndex(config.index_time_params)
                index.setQueryTimeParams(config.query_time_params)
                nbrs = index.knnQueryBatch(pc_bases[pseudo_c], k=config.K, num_threads=config.num_threads)
                for i in range(len(pc_idxs[pseudo_c])):
                    pseudo_idx = int(pc_idxs[pseudo_c][i])
                    dists[pseudo_idx] = float(nbrs[i][1][0])
        
        self.assessed_mlist.append(m_info)
        return dists