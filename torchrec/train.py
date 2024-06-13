import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim import Adam, SGD

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DNN')
logger.setLevel(logging.INFO)

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
    if optimizer is None:
        optimizer = model.configure_optimizers()
    if optimizer_scheduler is None and hasattr(model, 'configure_optimizer_scheduler'):
        optimizer_scheduler = model.configure_optimizer_scheduler(optimizer)

    eval_losses = []
    best_eval_loss = None

    for epoch in range(1, num_epochs+1):
        model.train()
        train_loss = {'total': 0., } or []
    
        # Training 
        # for k, (sparse_feats, dense_feats, labels) in enumerate(train_dataloader):
        for k, (features, labels) in enumerate(train_dataloader):
            # features = features.to(device)
            # labels = labels.to(device)
            
            optimizer.zero_grad()   # zero the parameter gradients
            loss = model.train_step(features, labels)

            # accumulate losses and compute gradients
            if isinstance(loss, dict):
                loss['total'].backward()
                train_loss = {n: train_loss.get(n,0) + v.item() for n, v in loss.items()}
            elif isinstance(loss, list):
                loss[0].backward()
                train_loss['total'] += loss[0].item()
                train_loss['details'] = []
                for k, v in enumerate( loss[1:] ):
                    if len(train_loss['details']) < k:
                        train_loss['details'].append(v.item())
                    else:
                        train_loss['details'][k] += v.item()
            else:
                loss.backward()
                train_loss['total'] += loss.item()
            
            optimizer.step()       # adjust parameters based on the calculated gradients 
            
            if k % 100 == 0:
                avg_loss = {n: v/k for n, v in train_loss.items()} if k > 0 else train_loss
                logger.info(f'[Training] Epoch: {epoch}/{num_epochs} iter {k}/{len(train_dataloader)}, Training Loss: {avg_loss}')
                
        optimizer_scheduler.step()
                
        for _type, _value in train_loss.items():
            train_loss[_type] = _value / len(train_dataloader)


        if eval_dataloader is None:
            continue

        # Validation
        with torch.no_grad(): 
            model.eval()
            eval_loss = {'total': 0.}

            # for sparse_feats, dense_feats, labels in eval_dataloader:             
            for features, labels in eval_dataloader:
                loss = model.eval_step(features, labels)

                if isinstance(loss, dict):
                    eval_loss = {n: eval_loss.get(n,0) + v.item() for n, v in loss.items()}
                elif isinstance(loss, list):
                    eval_loss['total'] += loss[0].item()
                    eval_loss['details'] = []
                    for k, v in enumerate( loss[1:] ):
                        if len(eval_loss['details']) < k:
                            eval_loss['details'].append(v.item())
                        else:
                            eval_loss['details'][k] += v.item()
                else:
                    eval_loss['total'] += loss.item()
                
            for _type, _value in eval_loss.items():
                eval_loss[_type] = _value / len(eval_dataloader)
                

        logger.info(f'[Validation] Epoch: {epoch}/{num_epochs}, Training Loss: {train_loss}, Validation Loss: {eval_loss}')

        if early_stopping_rounds:
            if len(eval_losses) >= early_stopping_rounds:
                eval_loss_his_avg = np.mean([v['total'] for v in eval_losses[-early_stopping_rounds:]])
                if eval_loss['total'] > eval_loss_his_avg:
                    logger.info(f'Early stopping at epoch {epoch}...')
                    break
        eval_losses.append(eval_loss)

        if best_model_path:
            if best_eval_loss is None or eval_loss['total'] < best_eval_loss:
                best_eval_loss = eval_loss['total']
                torch.save(model.state_dict(), best_model_path)

    if final_model_path:
        torch.save(model.state_dict(), final_model_path)
        
    if ret_model == 'best' and eval_dataloader is not None and best_model_path:
        model.load_state_dict(torch.load(best_model_path))

    return model