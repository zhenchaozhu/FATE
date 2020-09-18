import numpy as np


class EarlyStoppingCheckPoint(object):

    def __init__(self, learner, model, monitor_value, patience):
        self.learner = learner
        self.model = model
        self.monitor_value = monitor_value
        self.patience = patience
        self.wait = 0
        self.stopped_epoch = 0
        self.stopped_batch = 0
        self.stop_training = False
        self.best_metric = None
        self.cur_best_model = None

    def on_train_begin(self):
        self.wait = 0
        self.stopped_epoch = 0
        self.best_metric = -np.Inf

    def on_validation_end(self, curr_epoch, batch_idx, metrics_dict=None):

        current = metrics_dict.get(self.monitor_value)
        if current is None:
            print('monitored value does not available in the metrics dictionary.')
            return

        if current > self.best_metric:
            self.best_metric = current
            print("find best acc: ", self.best_metric, "at epoch:", curr_epoch, "batch:", batch_idx)
            best_model = self.model.export_model()
            self.cur_best_model = {'model': {'best_model': best_model}} if best_model is not None else None
            self.wait = 0
        else:
            self.wait += 1
            if self.wait > self.patience:
                print(f"early stop at epoch: {curr_epoch}, batch: {batch_idx}")
                self.stopped_epoch = curr_epoch
                self.stopped_batch = batch_idx
                self.stop_training = True

    def on_batch_end(self):
        return {'stop_training': self.stop_training}

    def on_epoch_end(self):
        return {'stop_training': self.stop_training}

    def on_train_end(self):
        if self.cur_best_model:
            self.learner.load_model(self.cur_best_model)
