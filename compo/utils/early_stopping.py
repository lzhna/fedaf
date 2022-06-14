from copy import deepcopy

class EarlyStopping:
    def __init__(self, patience=7, delta=0):
        self.patience = patience
        self.delta = delta
        self.min_loss = None
        self.early_stop = False
        self.checkpoint = None
        self.best_epoch = 0
        self.best_metrics = None
        
    def __call__(self, metrics, model, i_epoch):
        loss = metrics['loss'] / metrics['total']
        if self.min_loss is None:
            self.min_loss = loss
            self.save_checkpoint(metrics, model, i_epoch)
        elif loss > self.min_loss + self.delta:
            if (i_epoch - self.best_epoch) >= self.patience:
                self.early_stop = True
        else:
            self.min_loss = loss
            self.save_checkpoint(metrics, model, i_epoch)

    def save_checkpoint(self, metrics, model, i_epoch):
        '''Saves model when validation loss decrease.'''
        self.checkpoint = deepcopy(model)
        self.best_epoch = i_epoch
        self.best_metrics = metrics
    
    def get_checkpoint(self):
        return self.checkpoint, self.best_epoch, self.best_metrics
    
    def restart(self):
        self.early_stop = False
        self.best_epoch = 0
        self.checkpoint = None

