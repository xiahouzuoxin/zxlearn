import os
import numpy as np
import json

import torch
import torch.nn as nn
import torch.optim as optim

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Trainer')
logger.setLevel(logging.INFO)

class Trainer:
    def __init__(self, model: nn.Module, 
                 optimizer: optim.Optimizer = None, 
                 lr_scheduler=None, 
                 max_epochs=1, 
                 early_stopping_rounds=None, 
                 save_ckpt_path=None,
                 ckpt_prefix='checkpoint',
                 logger=logger,
                 **kwargs):
        """
        Initializes the Trainer object.

        Args:
            model (nn.Module): The neural network model.
            optimizer (optim.Optimizer, optional): The optimizer for training the model. If not provided, the model's `configure_optimizers` method will be used to configure the optimizer. Defaults to None.
            lr_scheduler (optional): The scheduler for the optimizer. If not provided and the model has a `configure_lr_scheduler` method, it will be used to configure the scheduler. Defaults to None.
            max_epochs (int, optional): The max number of training epochs. Defaults to 1.
            early_stopping_rounds (int, optional): The number of rounds to wait for early stopping. Defaults to None.
            save_ckpt_path (str, optional): The path to save the checkpoint files. Defaults to None.
            logger ([type], optional): The logger object. Defaults to logger.
        """
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.logger = logger

        self.default_ckpt_prefix = 'checkpoint' if ckpt_prefix is None else ckpt_prefix
        self.num_epoch = 0 
        self.global_steps = 0    

        if self.optimizer is None and hasattr(self.model, 'configure_optimizers'):
            self.optimizer = self.model.configure_optimizers()

        if self.lr_scheduler is None and hasattr(self.model, 'configure_lr_scheduler'):
            self.lr_scheduler = self.model.configure_lr_scheduler(self.optimizer)

        self.save_ckpt_path = save_ckpt_path
        if self.save_ckpt_path:
            # if os.path.exists(self.save_ckpt_path):
            #     # rename the existing directory
            #     os.rename(save_ckpt_path, f'{save_ckpt_path.rstrip('/')}.old')
            os.makedirs(self.save_ckpt_path, exist_ok=True)

            self.metadata_fn = f'{self.save_ckpt_path}/metadata.json'

        self.max_epochs = max_epochs
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

    def fit(self, 
            train_dataloader: torch.utils.data.DataLoader,
            eval_dataloader: torch.utils.data.DataLoader = None,
            init_ckpt_path: str = None, 
            init_ckpt_exclude_keys: list = None,
            ret_model='final'):
        """
        Trains the model using the provided training dataloader and evaluates it using the optional evaluation dataloader.

        Args:
            train_dataloader (torch.utils.data.DataLoader): The dataloader for training the model.
            eval_dataloader (torch.utils.data.DataLoader, optional): The dataloader for evaluating the model. Defaults to None.
            init_ckpt_path (str, optional): The path to the initial checkpoint path or the file name. Defaults to None.
            init_ckpt_exclude_keys (list, optional): The keys to exclude from the initial checkpoint file. Defaults to None.
            ret_model (str, optional): The type of model to return. Can be 'final' or 'best'. Defaults to 'final'.

        Returns:
            The trained model.

        """
        if init_ckpt_path:
            self.load_ckpt(init_ckpt_path, exclude_keys=init_ckpt_exclude_keys)

        eval_losses = []

        eval_loss = self.evaluate_model(self.model, eval_dataloader)
        self.logger.info(f'[Validation] Epoch: {self.num_epoch}/{self.max_epochs}, Validation Loss: {eval_loss}')

        while self.num_epoch < self.max_epochs:
            self.num_epoch += 1
            
            self.model.train()
            train_loss = {'loss': 0., } or []

            # print the latest learning rate of lr_scheduler
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
                if self.lr_scheduler:
                    self.lr_scheduler.step()
                self.global_steps += 1
                
                if k % 100 == 0:
                    avg_loss = {n: v/k for n, v in train_loss.items()} if k > 0 else train_loss
                    self.logger.info(f'[Training] Epoch: {self.num_epoch}/{self.max_epochs} iter {k}/{len(train_dataloader)}, Training Loss: {avg_loss}')
                    
            for _type, _value in train_loss.items():
                train_loss[_type] = _value / len(train_dataloader)

            eval_loss = self.evaluate_model(self.model, eval_dataloader)
            self.logger.info(f'[Validation] Epoch: {self.num_epoch}/{self.max_epochs}, Validation Loss: {eval_loss}')

            self.save_ckpt(self.default_ckpt_prefix, local_steps=(self.num_epoch-1)*len(train_dataloader)+k, eval_loss=eval_loss)

            if self.early_stopping_rounds:
                if len(eval_losses) >= self.early_stopping_rounds:
                    eval_loss_his_avg = np.mean([v['loss'] for v in eval_losses[-self.early_stopping_rounds:]])
                    if eval_loss['loss'] > eval_loss_his_avg:
                        self.logger.info(f'Early stopping at epoch {self.num_epoch}...')
                        break
            eval_losses.append(eval_loss)

        if ret_model == 'best':
            self.load_ckpt(self.save_ckpt_path, ret_best=True)

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
            'local_steps': local_steps,
            'eval_loss': eval_loss,
            'model': self.model,
            'optimizer': self.optimizer,
            'lr_scheduler': self.lr_scheduler,
            **kwargs
        }

        # save all the other serializable attributes in __init__ function if not exist in save_dict
        for k, v in self.__dict__.items():
            # check if the attribute is serializable
            if k not in state_dict and not callable(v) and not k.startswith('_'):
                state_dict[k] = v

        torch.save(state_dict, f'{self.save_ckpt_path}/{prefix}.{self.global_steps:06}.ckpt')

        def _update_metadata():
            # save metadata as json file
            if os.path.exists(self.metadata_fn):
                with open(self.metadata_fn, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {}

            if eval_loss is not None and eval_loss['loss'] < metadata.get('best_eval_loss', float('inf')):
                metadata['best_eval_loss'] = eval_loss['loss']
                metadata['best_eval_global_steps'] = self.global_steps
                metadata['best_ckpt'] = f'{prefix}.{self.global_steps:06}.ckpt'

            metadata.update({
                'global_steps': self.global_steps,
                'last_ckpt': f'{prefix}.{self.global_steps:06}.ckpt',
                'local_steps': local_steps,
                'eval_loss': eval_loss
            })
            
            with open(self.metadata_fn, 'w') as f:
                json.dump(metadata, f, indent=4)

        _update_metadata()

        self.logger.info(f'Checkpoint saved at {self.save_ckpt_path}/{prefix}.{self.global_steps:06}.ckpt')

    def load_ckpt(self, path: str=None, exclude_keys=None, ret_best=False, raise_error=True):
        '''
        Load the checkpoint file with the given path.
        It will overwrite the model, optimizer, lr_scheduler and global_steps.

        Args:
            path (str): The path to the checkpoint file or the directory.
            exclude_keys (list, optional): The keys to exclude from the checkpoint file. Defaults to None.
            ret_best (bool, optional): Whether to return the best checkpoint file. Defaults to False.
            raise_error (bool, optional): Whether to raise error if the checkpoint file is not found. Defaults to True.

        Returns:
            The checkpoint file.
        '''
        if path is None:
            return self.load_ckpt(self.save_ckpt_path, exclude_keys=exclude_keys, ret_best=ret_best, raise_error=False)
        else:
            if os.path.isfile(path):
                ckpt_file = path
            elif os.path.isdir(path) and os.path.exists(self.metadata_fn):
                # get checkpoint file from metadata
                with open(self.metadata_fn, 'r') as f:
                    metadata = json.load(f)
                    ckpt_file = metadata['best_ckpt'] if ret_best else metadata['last_ckpt']
                    ckpt_file = os.path.join(path, ckpt_file)
                    if not os.path.exists(ckpt_file):
                        if raise_error:
                            raise FileNotFoundError(f'Checkpoint file {ckpt_file} not found.')
                        else:
                            return None
            else:
                if raise_error:
                    raise FileNotFoundError(f'Checkpoint file {path} not found.')
                else:
                    return None
        
        ckpt = torch.load(ckpt_file)

        exclude_keys = set([]) if exclude_keys is None else set(exclude_keys) 
        
        # load all the serializable attributes in the checkpoint file
        for k, v in ckpt.items():
            if k in exclude_keys:
                continue
            elif k in ['model', ]:
                model = eval(f'self.{k}')
                if model is None:
                    model = v
                    continue

                # check the state_dict consistency
                model.load_state_dict(v.state_dict())
                self.logger.info(f'Loaded {k} state_dict from checkpoint.')

                # load the other attributes in the model
                for mk, mv in v.__dict__.items():
                    if mk not in model.__dict__ or callable(mv) or mk.startswith('_') or mk in ['state_dict',]:
                        continue
                    setattr(model, mk, mv)
                    self.logger.info(f'Loaded {k}.{mk} from checkpoint.')
            elif k in self.__dict__:
                self.__dict__[k] = v
                # print it if v is not long
                str_v = str(v) if len(str(v)) < 100 else (str(v)[:100] + '...')
                self.logger.info(f'Loaded {k} = {str_v} from checkpoint.')

        self.logger.info(f'Checkpoint loaded from {ckpt_file}.')

        return ckpt