import torch
from torch.nn import CrossEntropyLoss

def evaluation_cls(model, data_loader):
    model = model.cuda()
    model.eval()
    criterion = CrossEntropyLoss().cuda()

    metrics = {'correct': 0,'loss': 0,'total': 0}
    with torch.no_grad():
        for batch_idx, (idxs, inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            _, predicted = torch.max(outputs, -1)
            correct = predicted.eq(targets).sum()

            metrics['correct'] += correct.item()
            metrics['loss'] += loss.item() * targets.size(0)
            metrics['total'] += targets.size(0)

    return metrics


def params_avg(w_params):
    n_train_sum = 0
    for n_train, _ in w_params:
        n_train_sum += n_train
        
    averaged_params = w_params[0][1]
    for k in averaged_params.keys():
        for i in range(0, len(w_params)):
            n_train, model_params = w_params[i]
            w = n_train / n_train_sum
            if i == 0:
                averaged_params[k] = model_params[k] * w
            else:
                averaged_params[k] += model_params[k] * w
        
    return averaged_params

def avg_teachers_out(teacher_per_sample, teachers_dict, idxs, inputs):
    if teacher_per_sample is None:
        if len(teachers_dict) == 1:
            outputs_t = teachers_dict[list(teachers_dict.keys())[0]](inputs)
        else:
            output_list = []
            for teacher_idx in teachers_dict.keys():
                output_s = teachers_dict[teacher_idx](inputs)
                output_list.append(output_s)
            outputs_t = torch.mean(torch.stack(output_list), 0)
    else:
        curs_tmp = {teacher_idx:0 for teacher_idx in teachers_dict.keys()}
        sampleid_curs = {teacher_idx:{} for teacher_idx in teachers_dict.keys()}
        value_tmp = {}
        for i in range(len(idxs)):
            i_sample = int(idxs[i])
            for teacher_idx in teacher_per_sample[i_sample]:
                sampleid_curs[teacher_idx][i_sample] = curs_tmp[teacher_idx]
                curs_tmp[teacher_idx] += 1
                value_tmp.setdefault(teacher_idx, []).append(inputs[i])
                
        for teacher_idx in value_tmp.keys():
            aggr_inputs = torch.stack(value_tmp[teacher_idx])
            value_tmp[teacher_idx] = teachers_dict[teacher_idx](aggr_inputs)
                
        output_tmp = []
        for i in range(len(idxs)):
            i_sample = int(idxs[i])
            to_avg_outputs = []
            for teacher_idx in teacher_per_sample[i_sample]:
                curs = sampleid_curs[teacher_idx][i_sample]
                to_avg_outputs.append(value_tmp[teacher_idx][curs])
            output_tmp.append(torch.mean(torch.stack(to_avg_outputs), 0))
        outputs_t = torch.stack(output_tmp)
        
    return outputs_t


def merge_dists(dlists:list, n_publ):
    if len(dlists) == 1:
        return dlists

    merged_dists = {}
    for i_proxy in range(n_publ):
        sample_dists = []
        for i_view in range(len(dlists)):
            if i_proxy in dlists[i_view]:
                sample_dists.append(dlists[i_view][i_proxy])

        if len(sample_dists) > 0:
            merged_dists[i_proxy] = min(sample_dists)

    return [merged_dists]

def aggregate_estim(joint_estim:dict, n_proxy):
    for m_info in joint_estim.keys():
        joint_estim[m_info] = merge_dists(joint_estim[m_info], n_proxy)

def select_teachers(joint_estim:dict, n_proxy, mu):
    teachers_per_sample = {}
    hard_sample_idxs = []
    for i_proxy in range(n_proxy):
        # qualified teachers
        for m_info in joint_estim.keys():
            if i_proxy in joint_estim[m_info][0] and joint_estim[m_info][0][i_proxy] <= mu:
                teachers_per_sample.setdefault(i_proxy, []).append(m_info)

        # selected teachers
        if i_proxy not in teachers_per_sample:
            hard_sample_idxs.append(i_proxy)
            teachers_per_sample[i_proxy] = [m_info for m_info in joint_estim.keys()]
    
    return teachers_per_sample, hard_sample_idxs

def improper_rate(teachers_per_sample, hard_sample_idxs, n_publ, n_base, test_bigger:bool):
    num = 0
    for i_proxy in range(n_publ):
        if i_proxy in hard_sample_idxs:
            continue

        n_qualified_teachers = len(teachers_per_sample[i_proxy])
        case_too_many = test_bigger and (n_qualified_teachers > n_base)
        case_too_few = (not test_bigger) and (n_qualified_teachers < n_base)
        if case_too_many or case_too_few:
            num += 1
    
    return num / (n_publ - len(hard_sample_idxs))

def dynamic_adjust(teachers_per_sample, hard_sample_idxs, n_publ, n_base, test_bigger, joint_assessment, mu, p_adjust, omega, tune_patience):
    while True:
        p_improper = improper_rate(teachers_per_sample, hard_sample_idxs, n_publ, n_base, test_bigger)
        if (p_improper <= p_adjust) or (tune_patience <= 0):
            return teachers_per_sample, hard_sample_idxs, mu
        else:
            mu = mu * omega if test_bigger else mu / omega
            teachers_per_sample, hard_sample_idxs = select_teachers(joint_assessment, n_publ, mu)
            tune_patience -= 1

def adaptive_selection(joint_estim, n_proxy, mu, args):
    n_candidates = len(joint_estim)
    base_excs = n_candidates * args.r_excs
    base_indq = n_candidates * args.r_indq
    teachers_per_sample, hard_sample_idxs = select_teachers(joint_estim, n_proxy, mu)
    if n_proxy - len(hard_sample_idxs) >= 3:
        teachers_per_sample, hard_sample_idxs, mu = dynamic_adjust(teachers_per_sample, hard_sample_idxs, n_proxy, base_excs, True, joint_estim, mu, args.p_adjust, args.omega, args.tune_patience)
        teachers_per_sample, hard_sample_idxs, mu = dynamic_adjust(teachers_per_sample, hard_sample_idxs, n_proxy, base_indq, False, joint_estim, mu, args.p_adjust, args.omega, args.tune_patience)

    return teachers_per_sample, mu

def count_adoption(teachers_per_sample):
    teachers_adoption = {}
    for i_sample in teachers_per_sample.keys():
        for t_info in teachers_per_sample[i_sample]:
            teachers_adoption.setdefault(t_info, 0)
            teachers_adoption[t_info] += 1
            
    return teachers_adoption