import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim import Adam, SGD

import lightning as pl

import logging
import numpy as np
import torch
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DNN')
logger.setLevel(logging.INFO)

class Trainer:
    def __init__(self, model: nn.Module, 
                 optimizer: optim.Optimizer = None, 
                 optimizer_scheduler=None, 
                 num_epochs=1, 
                 early_stopping_rounds=None, 
                 best_model_path=None, 
                 final_model_path=None):
        """
        Initializes the Trainer object.

        Args:
            model (nn.Module): The neural network model.
            optimizer (optim.Optimizer, optional): The optimizer for training the model. If not provided, the model's `configure_optimizers` method will be used to configure the optimizer. Defaults to None.
            optimizer_scheduler (optional): The scheduler for the optimizer. If not provided and the model has a `configure_optimizer_scheduler` method, it will be used to configure the scheduler. Defaults to None.
            num_epochs (int, optional): The number of training epochs. Defaults to 1.
            early_stopping_rounds (int, optional): The number of rounds to wait for early stopping. Defaults to None.
            best_model_path (str, optional): The file path to save the best model. Defaults to None.
            final_model_path (str, optional): The file path to save the final model. Defaults to None.
        """
        self.model = model
        self.optimizer = optimizer
        self.optimizer_scheduler = optimizer_scheduler

        if self.optimizer is None:
            self.optimizer = self.model.configure_optimizers()

        if self.optimizer_scheduler is None and hasattr(self.model, 'configure_optimizer_scheduler'):
            self.optimizer_scheduler = self.model.configure_optimizer_scheduler(self.optimizer)

        self.num_epochs = num_epochs
        self.early_stopping_rounds = early_stopping_rounds
        self.best_model_path = best_model_path
        self.final_model_path = final_model_path

    def fit(self, train_dataloader: torch.utils.data.DataLoader, eval_dataloader: torch.utils.data.DataLoader = None, ret_model='final'):
        """
        Trains the model using the provided training dataloader and evaluates it using the optional evaluation dataloader.

        Args:
            train_dataloader (torch.utils.data.DataLoader): The dataloader for training the model.
            eval_dataloader (torch.utils.data.DataLoader, optional): The dataloader for evaluating the model. Defaults to None.
            ret_model (str, optional): The type of model to return. Can be 'final' or 'best'. Defaults to 'final'.

        Returns:
            The trained model.

        """
        eval_losses = []
        best_eval_loss = None

        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            train_loss = {'total': 0., } or []

            for k, (features, labels) in enumerate(train_dataloader):
                self.optimizer.zero_grad()  # zero the parameter gradients
                loss = self.model.train_step(features, labels)

                if isinstance(loss, dict):
                    loss['total'].backward()
                    train_loss = {n: train_loss.get(n, 0) + v.item() for n, v in loss.items()}
                elif isinstance(loss, list):
                    loss[0].backward()
                    train_loss['total'] += loss[0].item()
                    train_loss['details'] = []
                    for k, v in enumerate(loss[1:]):
                        if len(train_loss['details']) < k:
                            train_loss['details'].append(v.item())
                        else:
                            train_loss['details'][k] += v.item()
                else:
                    loss.backward()
                    train_loss['total'] += loss.item()

                self.optimizer.step()  # adjust parameters based on the calculated gradients

                if k % 100 == 0:
                    avg_loss = {n: v / k for n, v in train_loss.items()} if k > 0 else train_loss
                    logger.info(
                        f'[Training] Epoch: {epoch}/{self.num_epochs} iter {k}/{len(train_dataloader)}, Training Loss: {avg_loss}')

            if self.optimizer_scheduler:
                self.optimizer_scheduler.step()

            for _type, _value in train_loss.items():
                train_loss[_type] = _value / len(train_dataloader)

            if eval_dataloader is None:
                continue

            with torch.no_grad():
                self.model.eval()
                eval_loss = {'total': 0.}

                for features, labels in eval_dataloader:
                    loss = self.model.eval_step(features, labels)

                    if isinstance(loss, dict):
                        eval_loss = {n: eval_loss.get(n, 0) + v.item() for n, v in loss.items()}
                    elif isinstance(loss, list):
                        eval_loss['total'] += loss[0].item()
                        eval_loss['details'] = []
                        for k, v in enumerate(loss[1:]):
                            if len(eval_loss['details']) < k:
                                eval_loss['details'].append(v.item())
                            else:
                                eval_loss['details'][k] += v.item()
                    else:
                        eval_loss['total'] += loss.item()

                for _type, _value in eval_loss.items():
                    eval_loss[_type] = _value / len(eval_dataloader)

            logger.info(
                f'[Validation] Epoch: {epoch}/{self.num_epochs}, Training Loss: {train_loss}, Validation Loss: {eval_loss}')

            if self.early_stopping_rounds:
                if len(eval_losses) >= self.early_stopping_rounds:
                    eval_loss_his_avg = np.mean([v['total'] for v in eval_losses[-self.early_stopping_rounds:]])
                    if eval_loss['total'] > eval_loss_his_avg:
                        logger.info(f'Early stopping at epoch {epoch}...')
                        break
            eval_losses.append(eval_loss)

            if self.best_model_path:
                if best_eval_loss is None or eval_loss['total'] < best_eval_loss:
                    best_eval_loss = eval_loss['total']
                    torch.save(self.model.state_dict(), self.best_model_path)

        if self.final_model_path:
            torch.save(self.model.state_dict(), self.final_model_path)

        if ret_model == 'best' and eval_dataloader is not None and self.final_model_path:
            self.model.load_state_dict(torch.load(self.final_model_path))

        return self.model

def train_loop(model: nn.Module, 
    train_dataloader: torch.utils.data.DataLoader, 
    num_epochs: int,
    eval_dataloader: torch.utils.data.DataLoader = None,
    optimizer: optim.Optimizer = None,
    optimizer_scheduler = None,    
    early_stopping_rounds: int = None,
    best_model_path=None,
    final_model_path=None,
    ret_model='final'
):
    trainer = Trainer(model, optimizer, optimizer_scheduler, num_epochs, early_stopping_rounds, best_model_path, final_model_path)
    return trainer.fit(train_dataloader, eval_dataloader, ret_model)
