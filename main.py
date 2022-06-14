from compo.utils.parameters import args_parser
from compo.utils.setup import *
from compo.fed.load_dataset import load_fed_data, NC
from compo.fed.client import Client
from compo.fed.model_trainer import ModelTrainer
from compo.fed.ops import *
from torch.utils.data import DataLoader
from torch.multiprocessing import Pool, set_start_method
from copy import deepcopy
from datetime import datetime


def local_steps(trainer:ModelTrainer, gmodel, client_idx, tloader, i_iter):
    init_seed(i_iter)
    trainer.train_cls(gmodel, tloader)
    gmodel = gmodel.cpu()
    return client_idx, gmodel.state_dict()

def main(args, root_path):
    proxy_set, gval_set, cpriv_sets, gtest_set, label_distr = load_fed_data(args, gd_dir='pdmap')
    gmodel = create_model(args.arch, NC[args.priv_ds]).cuda()
    trainer =ModelTrainer(args)

    # Build clients
    config = load_knn_config(args) if args.expr_mode == 'fedaf' else None
    aux_c = Config()
    aux_c.config = config
    client_list = []
    for i in range(args.n_clients):
        aux = deepcopy(aux_c)
        aux.local_label = list(label_distr[i].keys())
        client = Client(args=args,
                        priv_ds=cpriv_sets[i],
                        device_idx=1,
                        aux=aux)
        client_list.append(client)
    
    # Initialization
    mlog_fpath = os.path.join(root_path, 'gmodel.csv')
    mlog_title = 'i_iter,epochs,dist_ceil,remark,gacc,gloss'
    mlogger = MetricsLogger(mlog_fpath, mlog_title)
    gtest_loader = DataLoader(gtest_set, batch_size=args.static_bs, shuffle=False)
    if args.expr_mode != 'fedavg':
        label_list = range(NC[args.priv_ds])
        n_proxy = len(proxy_set)
        ploader = DataLoader(proxy_set, batch_size=args.skd_bs, shuffle=True)
        ploader_fixed = DataLoader(proxy_set, batch_size=args.static_bs, shuffle=False)
        vloader = DataLoader(gval_set, batch_size=args.static_bs, shuffle=False)
    
    ## Multiprocessing env
    if args.pool_size > 1:
        set_start_method('spawn')

    ## Initialization for FedAF
    n_least_assessed = round(args.n_clients * args.r_least_estim)
    mu = args.mu
    comp_models, newly_updated = {}, {}
    joint_estim, feats_on_proxy, estim_states = {}, {}, {}

    # FL training
    trend = Trend(patience=args.expr_patience, higher=True)
    stop_iter = args.n_iter
    reviewer_cidxs = []

    for i_iter in range(1, args.n_iter+1):
        start = datetime.now()
        active_cidxs = client_sampling(i_iter, args.n_clients, args.r_active)

        if args.expr_mode == 'fedaf':
            # Sample reviewers
            if args.r_reviewer <= 0:
                reviewer_cidxs = active_cidxs
            elif args.r_reviewer < 1.0:
                reviewer_cidxs = client_sampling(i_iter+1, args.n_clients, args.r_reviewer)
            else:
                reviewer_cidxs = range(args.n_clients)
        
            # Aggregate global models too
            if not args.ab_global:
                comp_models[(-1, i_iter-1)] = deepcopy(gmodel)
                pc_bases, pc_idxs = trainer.get_cls_bases(model=gmodel,
                                                data_loader=ploader_fixed,
                                                correct_only=False,
                                                label_list=label_list,
                                                T=args.dist_T)
                feats_on_proxy[(-1, i_iter-1)] = (pc_bases, pc_idxs)
        
        # Local training
        start_training = datetime.now()
        p = Pool(args.pool_size)
        stepped = []
        for client_idx in active_cidxs:
            res = p.apply_async(func=local_steps,
                                args=(trainer,
                                    deepcopy(gmodel),
                                    client_idx,
                                    client_list[client_idx].tloader,
                                    i_iter))
            stepped.append(res)
        p.close()
        p.join()

        updated_models = {}
        for res in stepped:
            client_idx, model_params = res.get()
            updated_models[client_idx] = model_params
        
        p.terminate()
        end_training = datetime.now()

        w_param_fedavg = []
        teachers_dict = {}
        for client_idx in range(args.n_clients):
            client:Client = client_list[client_idx]
            if client_idx in active_cidxs:
                model_params = updated_models[client_idx]
                w_param_fedavg.append((client.n_train, model_params))

                if args.expr_mode == 'feddf':
                    teachers_dict[client_idx] = deepcopy(gmodel)
                    teachers_dict[client_idx].load_state_dict(model_params)
                elif args.expr_mode == 'fedaf':
                    cmodel = deepcopy(gmodel)
                    cmodel.load_state_dict(model_params)
                    newly_updated[(client_idx, i_iter)] = cmodel

                    pc_bases, pc_idxs = trainer.get_cls_bases(model=cmodel,
                                                data_loader=ploader_fixed,
                                                correct_only=False,
                                                label_list=label_list,
                                                T=args.dist_T)
                    feats_on_proxy[(client_idx, i_iter)] = (pc_bases, pc_idxs)
                    dists = client.estim(trainer, cmodel, pc_bases, pc_idxs, (client_idx, i_iter))
                    joint_estim.setdefault((client_idx, i_iter), []).append(dists)
                    estim_states[(client_idx, i_iter)] = 1

            # Jointly estimate models' fitness
            if client_idx in reviewer_cidxs:
                for m_info in comp_models.keys():
                    if m_info in client.assessed_mlist:
                        continue
                    
                    tmodel = comp_models[m_info]
                    pc_bases, pc_idxs = feats_on_proxy[m_info]
                    dists = client.estim(trainer, tmodel, pc_bases, pc_idxs, m_info)
                    joint_estim.setdefault(m_info, []).append(dists)
                    estim_states.setdefault(m_info, 0)
                    estim_states[m_info] += 1
        
        end_estim = datetime.now()

        # Adaptive teacher selection
        if args.expr_mode == 'fedaf':
            for m_info in newly_updated.keys():
                comp_models[m_info] = newly_updated[m_info]
            newly_updated.clear()
            aggregate_estim(joint_estim, n_proxy)
            teachers_per_sample, mu = adaptive_selection(joint_estim, n_proxy, mu, args)
            teachers_adoption = count_adoption(teachers_per_sample)
            
            w_param_adpt = []
            for m_info in teachers_adoption:
                tmp = deepcopy(comp_models[m_info])
                w_param_adpt.append((teachers_adoption[m_info], tmp.state_dict()))
        
        end_selection = datetime.now()

        # Parameters averaging
        averaged_params = params_avg(w_param_fedavg) if args.ab_cwpa else params_avg(w_param_adpt)
        gmodel.load_state_dict(averaged_params)
        metrics = evaluation_cls(gmodel, gtest_loader)
        gacc = metrics['correct'] / metrics['total']
        gloss = metrics['loss'] / metrics['total']
        remark = 'fedavg' if args.ab_cwpa else 'cwpa'
        mlogger.log(f'{i_iter},{args.ltrain_epoch},nah,{remark},{gacc},{gloss}')

        # Ensemble distillation
        if not args.ab_ens_kd:
            if args.ab_kf:
                teachers_per_sample = None
            
            if args.expr_mode == 'fedaf':
                teachers_dict = {m_info: comp_models[m_info]
                                for m_info in teachers_adoption}
            
            init_seed(i_iter)
            gmodel, stop_batch = trainer.distill(gmodel, teachers_dict, ploader, vloader, teachers_per_sample)
            if stop_batch > 0:
                metrics = evaluation_cls(gmodel, gtest_loader)
                gacc = metrics['correct'] / metrics['total']
                gloss = metrics['loss'] / metrics['total']
                
            remark = 'skd[equ_contri]' if args.ab_kf else 'skd[adoption]'
            remark_dceil = 'nah' if args.ab_kf else f'{mu}'
            mlogger.log(f'{i_iter},{stop_batch},{remark_dceil},{remark},{gacc},{gloss}')

        end_ens = datetime.now()

        if args.expr_mode == 'fedaf':
            # Remove fully used
            ms = list(estim_states.keys())
            for m_info in ms:
                if estim_states[m_info] >= n_least_assessed:
                    comp_models.pop(m_info)
                    joint_estim.pop(m_info)
                    feats_on_proxy.pop(m_info)
                    estim_states.pop(m_info)

        end = datetime.now()
        reach_best = trend(gacc, i_iter)
        if i_iter % args.plog_freq == 0:
            mlogger.print()

        if i_iter % args.print_freq == 0:
            logging.debug(f'after {i_iter} iters, best gacc={trend.best_value} at i_iter={trend.best_idnt}, current iter costs: {end - start}')
            
            if args.expr_mode == 'fedaf' and args.analysis:
                logging.debug(f'|- local training: {end_training - start_training}')
                logging.debug(f'|- estim: {end_estim - end_training}')
                logging.debug(f'|- teacher selection: {end_selection - end_estim}')
                logging.debug(f'|- model fusion: {end_ens - end_selection}')
                logging.debug(f'|- remove fully used: {end - end_ens}')
            
        if i_iter > args.expr_least and reach_best:
            stop_iter = i_iter
            break
        
    logging.debug(f'finished {stop_iter} iters, best global_acc={trend.best_value} at i_iter={trend.best_idnt}')

if __name__ == '__main__':
    args = args_parser()
    args = workflow(args)
    init_seed(args.seed)
    root_path, timestamp = create_log_dir(args.expr_mode)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    logging.debug(f'expr_mode = {args.expr_mode}, timestamp = {timestamp}, PID = {os.getpid()}, gpu={args.gpu}')
    logging.debug(f'args = {args}')

    main(args, root_path)