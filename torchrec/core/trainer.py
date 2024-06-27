import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# import lightning as pl

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
                 save_ckpt_path=None,
                 init_ckpt_path=None,
                 logger=logger,
                 **kwargs):
        """
        Initializes the Trainer object.

        Args:
            model (nn.Module): The neural network model.
            optimizer (optim.Optimizer, optional): The optimizer for training the model. If not provided, the model's `configure_optimizers` method will be used to configure the optimizer. Defaults to None.
            optimizer_scheduler (optional): The scheduler for the optimizer. If not provided and the model has a `configure_optimizer_scheduler` method, it will be used to configure the scheduler. Defaults to None.
            num_epochs (int, optional): The number of training epochs. Defaults to 1.
            early_stopping_rounds (int, optional): The number of rounds to wait for early stopping. Defaults to None.
            save_ckpt_path (str, optional): The path to save the checkpoint files. Defaults to None.
            init_ckpt_path (str, optional): The path to the initial checkpoint path or the file name. Defaults to None.
            logger ([type], optional): The logger object. Defaults to logger.
        """
        self.model = model
        self.optimizer = optimizer
        self.optimizer_scheduler = optimizer_scheduler
        self.logger = logger

        self.default_ckpt_prefix = 'checkpoint'    
        self.global_steps = 0    

        if self.optimizer is None:
            self.optimizer = self.model.configure_optimizers()

        if self.optimizer_scheduler is None and hasattr(self.model, 'configure_optimizer_scheduler'):
            self.optimizer_scheduler = self.model.configure_optimizer_scheduler(self.optimizer)

        self.save_ckpt_path = save_ckpt_path
        if self.save_ckpt_path:
            if not os.path.exists(self.save_ckpt_path):
                os.makedirs(self.save_ckpt_path)

        if init_ckpt_path:
            ckpt = self.load_ckpt(init_ckpt_path)
            self.global_steps = ckpt.get('global_steps', 0)

        self.num_epochs = num_epochs
        self.early_stopping_rounds = early_stopping_rounds

        # all kwargs are saved as attributes
        for k, v in kwargs.items():
            setattr(self, k, v)

    def evaluate_model(self, model, eval_dataloader):
        if eval_dataloader is None:
            return None
        
        with torch.no_grad(): 
            model.eval()
            eval_loss = {'loss': 0.}

            for k, batch in enumerate(eval_dataloader):
                loss = model.validation_step(batch, k)

                if isinstance(loss, dict):
                    eval_loss = {n: eval_loss.get(n,0) + v.item() for n, v in loss.items()}
                elif isinstance(loss, list):
                    eval_loss['loss'] += loss[0].item()
                    eval_loss['details'] = []
                    for k, v in enumerate( loss[1:] ):
                        if len(eval_loss['details']) < k:
                            eval_loss['details'].append(v.item())
                        else:
                            eval_loss['details'][k] += v.item()
                else:
                    eval_loss['loss'] += loss.item()

            for _type, _value in eval_loss.items():
                eval_loss[_type] = _value / len(eval_dataloader)

        return eval_loss

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

        eval_loss = self.evaluate_model(self.model, eval_dataloader)
        self.logger.info(f'[Validation] Epoch: 0/{self.num_epochs}, Validation Loss: {eval_loss}')

        for epoch in range(1, self.num_epochs+1):
            self.model.train()
            train_loss = {'loss': 0., } or []

            # print the latest learning rate of optimizer_scheduler
            self.logger.info(f'Learning rate: {self.optimizer.param_groups[0]["lr"]}') # the learning rate of the first parameter group
        
            # Training 
            for k, batch in enumerate(train_dataloader):
                self.optimizer.zero_grad()   # zero the parameter gradients
                loss = self.model.training_step(batch, k)

                # accumulate losses and compute gradients
                if isinstance(loss, dict):
                    loss['loss'].backward()
                    train_loss = {n: train_loss.get(n,0) + v.item() for n, v in loss.items()}
                elif isinstance(loss, list):
                    loss[0].backward()
                    train_loss['loss'] += loss[0].item()
                    train_loss['details'] = []
                    for k, v in enumerate( loss[1:] ):
                        if len(train_loss['details']) < k:
                            train_loss['details'].append(v.item())
                        else:
                            train_loss['details'][k] += v.item()
                else:
                    loss.backward()
                    train_loss['loss'] += loss.item()
                
                self.optimizer.step()       # adjust parameters based on the calculated gradients 
                if self.optimizer_scheduler:
                    self.optimizer_scheduler.step()
                self.global_steps += 1
                
                if k % 100 == 0:
                    avg_loss = {n: v/k for n, v in train_loss.items()} if k > 0 else train_loss
                    self.logger.info(f'[Training] Epoch: {epoch}/{self.num_epochs} iter {k}/{len(train_dataloader)}, Training Loss: {avg_loss}')
                    
            for _type, _value in train_loss.items():
                train_loss[_type] = _value / len(train_dataloader)

            eval_loss = self.evaluate_model(self.model, eval_dataloader)
            self.logger.info(f'[Validation] Epoch: {epoch}/{self.num_epochs}, Validation Loss: {eval_loss}')

            if best_eval_loss is None or eval_loss['loss'] < best_eval_loss:
                best_eval_loss = eval_loss['loss']
                ckpt_prefix = f'{self.default_ckpt_prefix}.best'
            else:
                ckpt_prefix = self.default_ckpt_prefix
            self.save_ckpt(ckpt_prefix, local_steps=epoch*len(train_dataloader)+k, eval_loss=eval_loss, best_eval_loss=best_eval_loss)

            if self.early_stopping_rounds:
                if len(eval_losses) >= self.early_stopping_rounds:
                    eval_loss_his_avg = np.mean([v['loss'] for v in eval_losses[-self.early_stopping_rounds:]])
                    if eval_loss['loss'] > eval_loss_his_avg:
                        self.logger.info(f'Early stopping at epoch {epoch}...')
                        break
            eval_losses.append(eval_loss)
            
        if ret_model == 'best' and eval_dataloader is not None and self.save_ckpt_path:
            self.load_ckpt(f'{self.default_ckpt_prefix}.best')

        return self.model
    
    def save_ckpt(self, prefix: str, local_steps: int = None, eval_loss=None, **kwargs):
        '''
        Save the checkpoint file with the given prefix and epoch number.

        Args:
            prefix (str): The prefix of the checkpoint file name.
            epoch (int, optional): The epoch number. Defaults to None.
            local_steps (int, optional): The local steps. Defaults to None.
            eval_loss ([type], optional): The evaluation loss. Defaults to None.
        '''
        if self.save_ckpt_path is None:
            return

        state_dict = {
            'global_steps': self.global_steps,
            'local_steps': local_steps,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'optimizer_scheduler_state_dict': self.optimizer_scheduler.state_dict() if self.optimizer_scheduler else None,
            'eval_loss': eval_loss,
            'feat_configs': self.model.feat_configs if hasattr(self.model, 'feat_configs') else None
            **kwargs
        }

        # save all the other serializable attributes in __init__ function if not exist in save_dict
        for k, v in self.__dict__.items():
            # check if the attribute is serializable
            if k not in state_dict and not callable(v) and not isinstance(v, (nn.Module, optim.Optimizer, lr_scheduler._LRScheduler)):
                state_dict[k] = v

        torch.save(state_dict, f'{self.save_ckpt_path}/{prefix}.{self.global_steps}.ckpt')

        self.logger.info(f'Checkpoint saved at {self.save_ckpt_path}/{prefix}.{self.global_steps}.ckpt')

    def load_ckpt(self, prefix_or_path: str):
        '''
        Load the checkpoint file with the given prefix.

        Args:
            prefix (str): The prefix of the checkpoint file name 

        Returns:
            The checkpoint file.
        '''
        # check if prefix is a file not a directory

        # ckpt_path = prefix_or_path or self.save_ckpt_path
        if os.path.isfile(prefix_or_path):
            ckpt_file = prefix_or_path
        else:
            if os.path.isdir(prefix_or_path):
                ckpt_path = prefix_or_path
                prefix = self.default_ckpt_prefix
            elif self.save_ckpt_path:
                ckpt_path = self.save_ckpt_path
                prefix = prefix_or_path
            else:
                raise ValueError(f'Invalid prefix_or_path: {prefix_or_path}')

            # get the checkpoint file name with the max global steps
            ckpt_files = [f for f in os.listdir(ckpt_path) if f.startswith(prefix)]
            if len(ckpt_files) == 0:
                raise FileNotFoundError(f'No checkpoint files found in "{ckpt_path}" that with prefix "{prefix}".')
            global_steps = [int(f.split('.')[-2]) for f in ckpt_files]
            fname = ckpt_files[np.argmax(global_steps)]
            ckpt_file = f'{ckpt_path}/{fname}'
        
        ckpt = torch.load(ckpt_file)
        self.global_steps = ckpt['global_steps']
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if self.optimizer_scheduler:
            self.optimizer_scheduler.load_state_dict(ckpt['optimizer_scheduler_state_dict'])
        
        # load all the other serializable attributes in __init__ function if exist in ckpt
        # for k, v in self.__dict__.items():
        #     if k in ckpt:
        #         setattr(self, k, ckpt[k])

        self.logger.info(f'Checkpoint loaded from {ckpt_file}.')

        return ckpt