import logging
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from compo.fed.ops import avg_teachers_out, evaluation_cls
from compo.utils.early_stopping import EarlyStopping

class SoftTarget(nn.Module):
	def __init__(self, T):
		super(SoftTarget, self).__init__()
		self.T = T

	def forward(self, out_s, out_t):
		loss = F.kl_div(F.log_softmax(out_s/self.T, dim=1),
						F.softmax(out_t/self.T, dim=1),
						reduction='batchmean') * self.T * self.T

		return loss

OPTIM = {
    'adam':Adam
}

class ModelTrainer:
    def __init__(self, args):
        self.args = args
    
    def create_optimizer(self, model, loc):
        if loc == 'train_cls':
            optim = OPTIM[self.args.ltrain_optim]
            lr = self.args.ltrain_lr
        elif loc == 'distill':
            optim = OPTIM[self.args.skd_optim]
            lr = self.args.skd_lr
        else:
            exit(f'not support {loc}')

        return optim(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    def train_cls(self, model, tloader):
        model = model.cuda()
        model.train()

        criterion = CrossEntropyLoss().cuda()
        optimizer = self.create_optimizer(model, loc='train_cls')

        for _ in range(1, self.args.ltrain_epoch + 1):
            for _, (_, inputs, targets) in enumerate(tloader):
                inputs, targets = inputs.cuda(), targets.cuda()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 5.0)
                optimizer.step()
    
    def train_es(self, model, tloader, vloader, epoch, eval_freq, patience):
        model = model.cuda()
        model.train()

        criterion = CrossEntropyLoss().cuda()
        optimizer = self.create_optimizer(model, loc='train_cls')
        early_stopping = EarlyStopping(patience)
        metrics = evaluation_cls(model, vloader)
        early_stopping(metrics, model, 0)

        for i_epoch in range(1, epoch + 1):
            for _, (_, inputs, targets) in enumerate(tloader):
                inputs, targets = inputs.cuda(), targets.cuda()
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 5.0)
                optimizer.step()

            if i_epoch % eval_freq == 0:
                metrics = evaluation_cls(model, vloader)
                model.train()
                early_stopping(metrics, model, i_epoch)
                if early_stopping.early_stop:
                    break
        
        if early_stopping.early_stop:
            model, stop_epoch, metrics = early_stopping.get_checkpoint()
        else:
            stop_epoch = epoch
        
        return model, stop_epoch, metrics

    def get_cls_bases(self, model, data_loader, correct_only, label_list=None, T=None):
        model = model.cuda()
        model.eval()
        c_bases, c_idxs = {}, {}

        with torch.no_grad():
            for i_batch, iit in enumerate(data_loader):
                idxs = iit[0].cuda()
                inputs = iit[1].cuda()
                feats, outputs = model(inputs, feats_also=True)
                _, predicted = torch.max(outputs, -1)
                batch_size = inputs.size(0)

                if T is None:
                    bases = F.normalize(feats, dim=1)
                else:
                    bases = F.softmax(outputs / T, dim=1)
                
                if correct_only:
                    targets = iit[2].cuda()
                    correct_states = predicted.eq(targets)

                for c in label_list:
                    c_targets = torch.ones(batch_size).long().cuda() * c
                    cwatch = predicted.eq(c_targets)
                    if correct_only:
                        cwatch = cwatch * correct_states
                    
                    posi_idxs = cwatch.nonzero().view(-1)
                    if posi_idxs.size(0) > 0:
                        c_idxs.setdefault(c, []).append(idxs.index_select(dim=0, index=posi_idxs))
                        c_bases.setdefault(c, []).append(bases.index_select(dim=0, index=posi_idxs))

        for c in c_bases.keys():
            c_bases[c] = torch.cat(c_bases[c], dim=0).cpu()
            c_idxs[c] = torch.cat(c_idxs[c], dim=0).cpu()

        return c_bases, c_idxs
    
    def distill(self, model, teachers_dict, ploader, vloader, teacher_per_sample=None):
        for teacher_idx in teachers_dict.keys():
            teachers_dict[teacher_idx] = teachers_dict[teacher_idx].cuda()
            teachers_dict[teacher_idx].eval()
        
        t_outputs_dict = {}
        with torch.no_grad():
            for i_batch, iits in enumerate(ploader):
                idxs = iits[0]
                inputs = iits[1].cuda()
                # logging.debug(f'inputs.shape = {inputs.shape}')
                outputs_t = avg_teachers_out(teacher_per_sample, teachers_dict, idxs, inputs).cpu().detach()
                for i in range(len(idxs)):
                    sample_idx = int(idxs[i])
                    t_outputs_dict[sample_idx] = outputs_t[i]
            
        model = model.cuda()
        model.train()
        criterion = SoftTarget(self.args.skd_T)
        optimizer = self.create_optimizer(model, loc='distill')
        scheduler = CosineAnnealingLR(optimizer, self.args.skd_batch)
        early_stopping = EarlyStopping(self.args.skd_patience)
        metrics = evaluation_cls(model, vloader)
        early_stopping(metrics, model, 0)

        data_iter = iter(ploader)
        for i_batch in range(1, self.args.skd_batch + 1):
            try:
                idxs, inputs = next(data_iter)
            except StopIteration:
                data_iter = iter(ploader)
                idxs, inputs = next(data_iter)
            
            inputs = inputs.cuda()
            optimizer.zero_grad()

            outputs = model(inputs)
            outputs_list = []
            for i in range(len(idxs)):
                sample_idx = int(idxs[i])
                outputs_list.append(t_outputs_dict[sample_idx])
            outputs_t = torch.stack(outputs_list).cuda()

            loss = criterion(outputs, outputs_t)
            loss.backward()
            clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 5.0)
            optimizer.step()
            scheduler.step()

            if i_batch % self.args.skd_eval_freq == 0:
                metrics = evaluation_cls(model, vloader)
                model.train()
                early_stopping(metrics, model, i_batch)
                if early_stopping.early_stop:
                    break
        
        if early_stopping.early_stop:
            model, stop_batch, _ = early_stopping.get_checkpoint()
        else:
            stop_batch = self.args.skd_batch
        
        return model, stop_batch