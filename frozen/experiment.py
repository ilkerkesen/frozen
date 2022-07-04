import torch
import torch.nn as nn
from torch import optim
import pytorch_lightning as pl
from .model import OPTCaptioningModel


class Experiment(pl.LightningModule):
    def __init__(self, config=dict()):
        super().__init__()
        self.config = config
        self.model = OPTCaptioningModel(config.get('model', dict()))
        self.loss_fn = nn.CrossEntropyLoss()
        self.save_hyperparameters(config)

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def training_step(self, batch, batch_index):
        mask = batch['image_token_mask'].nonzero(as_tuple=True)
        labels = batch['input_ids'].clone()
        labels[mask] = -100  # default ignore index

        kwargs = {
            'pixel_values': batch['pixel_values'],
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
            'image_token_mask': batch['image_token_mask'],
            'labels': labels,
        }

        output = self.forward(**kwargs)
        return {'loss': output.loss}

    @torch.no_grad()
    def validation_step(self, batch, batch_index):
        return self.training_step(batch, batch_index)

    @property
    def optimizer(self):
        default = {
            'algorithm': 'Adam',
            'params': {
                'lr': 0.0003,
                'betas': [0.9, 0.95],
            },
        }

        return self.config.get('optimizer', default)


    def configure_optimizers(self):
        method = eval(f"optim.{self.optimizer['algorithm']}")
        params = self.optimizer['params']

        parameters = list()
        for child in self.model.children():
            lr = params.get('lr')
            parameters.append({'params': child.parameters(), 'lr': lr})

        optimizer = method(parameters, **params)

        return {
            'optimizer': optimizer,
        }       