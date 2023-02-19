import numpy as np
import torch

class EarlyStopping:

    def __init__(self, model_path, patience=5, verbose=True):
        """
        :param model_path: str, where to save model, not mode
        :param patience: int, how long to wait after last time validatioin loss improved
                                Default: 5
        :param verbose: bool if True, prints a message for each validation loss improvement.
                            Default: True
        """

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.model_path = model_path

    def __call__(self, val_loss, model):

        score = -val_loss
        print("Score!", score)
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print('Validation loss decreased ({} --> {}).  Saving model ...{}'.format(self.val_loss_min, val_loss,
                                                                                          self.model_path))
            #torch.save(model, self.model_path)
            self.val_loss_min = val_loss