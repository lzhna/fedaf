import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='resnet8')
    parser.add_argument('--priv_ds', type=str, default='cifar10')
    parser.add_argument('--publ_ds', type=str, default='cifar10')
    parser.add_argument('--r_proxy', type=float, default=0.2)
    parser.add_argument('--r_gval', type=float, default=0.08)
    parser.add_argument('--sub_proxy', type=float, default=None)
    parser.add_argument('--np', type=int, default=50000)

    parser.add_argument('--use_norm', action='store_true')
    parser.add_argument('--use_aug', action='store_true')
    parser.add_argument('--dsize', type=int, default=None)
    
    parser.add_argument('--n_clients', type=int, default=20)
    parser.add_argument('--r_active', type=float, default=0.5)
    parser.add_argument('--d_alpha', type=float, default=0.1)

    parser.add_argument('--ltrain_bs', type=int, default=32)
    parser.add_argument("--ltrain_epoch", type=int, default=40)
    parser.add_argument('--ltrain_lr', type=float, default=0.01)
    parser.add_argument('--ltrain_optim', type=str, default='adam')

    parser.add_argument('--skd_bs', type=int, default=128)
    parser.add_argument('--skd_batch', type=int, default=10000)
    parser.add_argument('--skd_lr', type=float, default=0.001)
    parser.add_argument('--skd_T', type=float, default=4.0)
    parser.add_argument('--skd_patience', type=int, default=1000)
    parser.add_argument('--skd_eval_freq', type=int, default=100)
    parser.add_argument('--skd_optim', type=str, default='adam')

    parser.add_argument('--r_least_estim', type=float, default=1.0)
    parser.add_argument('--r_reviewer', type=float, default=1.0)

    parser.add_argument('--mu', type=float, default=0.06)
    parser.add_argument('--dist_T', type=float, default=None)
    parser.add_argument('--r_excs', type=float, default=0.67)
    parser.add_argument('--r_indq', type=float, default=0.33)
    parser.add_argument('--p_adjust', type=float, default=0.67)
    parser.add_argument('--omega', type=float, default=0.95)
    parser.add_argument('--tune_patience', type=int, default=50000)

    parser.add_argument("--n_iter", type=int, default=10000)
    parser.add_argument('--data_seed', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--analysis', action='store_true')
    parser.add_argument("--print_freq", type=int, default=25)
    parser.add_argument("--plog_freq", type=int, default=25)
    parser.add_argument("--expr_patience", type=int, default=200)
    parser.add_argument("--expr_least", type=int, default=100)
    parser.add_argument("--static_bs", type=int, default=2048)
    parser.add_argument('--pool_size', type=int, default=5)
    parser.add_argument('--trans_params', action='store_true')

    parser.add_argument('--ab_global', action='store_true', help='w/o ensemble globel models')
    parser.add_argument('--ab_ens_kd', action='store_true', help='w/o ensemble distillation')
    parser.add_argument('--ab_cwpa', action='store_true', help='initialized by FedAvg')
    parser.add_argument('--ab_kf', action='store_true', help='w/o knowledge filtering')
    parser.add_argument('--ab_repr', action='store_true', help='using logits')

    parser.add_argument('--expr_mode', type=str, default='fedaf')
    args = parser.parse_args()
    return args