# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>,
# License: TDG-Attribution-NonCommercial-NoDistrib


import glob
import importlib
import yaml
import sys
import os
import re
from datetime import datetime
import shutil

import torch
import torch.optim as optim
import timm


def print_tree(d, indent=0, logger=None):
    for key, val in d.items():

        if isinstance(val, dict):
            if logger:
                logger.info(' ' * indent + str(key) + ': {')
            else:
                print(' ' * indent + str(key) + ': {')
            print_tree(val, indent + 4, logger)
            if logger:
                logger.info(' ' * indent + '}')
            else:
                print(' ' * indent + '}')
        else:
            if logger:
                logger.info(' ' * indent + str(key) + ': ' + str(val)) 
            else:
                print(' ' * indent + str(key) + ': ' + str(val))  

                
def load_saved_model(saved_path, model, ckpt, logger=None):
    """
    Load saved model if exiseted

    Parameters
    __________
    saved_path : str
       model saved path
    model : opencood object
        The model instance.

    Returns
    -------
    model : opencood object
        The model instance loaded pretrained params.
    """
    saved_path = saved_path + '/ckpt/'
    assert os.path.exists(saved_path), '{} not found'.format(saved_path)

    model_file = os.path.join(saved_path, 'net_epoch%d.pth' % ckpt)
    
    if not os.path.exists(model_file):
        if logger == None:
            print("{} is not exist.".format(model_file))
        else:
            logger.info("{} is not exist.".format(model_file))
        return False, model
    else:
        if logger == None:
            print('resuming by loading epoch %d' % ckpt)
        else:
            logger.info('resuming by loading epoch %d' % ckpt)
        checkpoint = torch.load(
            model_file,
            map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)

        del checkpoint

    return True, model


def setup_train(hypes, hypes_yaml):
    """
    Create folder for saved model based on current timestep and model name

    Parameters
    ----------
    hypes: dict
        Config yaml dictionary for training:
    """
    rank = int(os.getenv("RANK", 0))

    model_name = hypes['name']
    current_dir = os.getcwd()

    group_path = '/'.join(hypes_yaml.split('/')[3:-1])

    output_path = os.path.abspath(current_dir + "/../../logs/")

    full_path = os.path.join(output_path , group_path, model_name)

    if rank == 0:
        if not os.path.exists(full_path):
            if not os.path.exists(full_path):
                try:
                    os.makedirs(full_path)
                except FileExistsError:
                    pass
            # save the yaml file
            save_name = os.path.join(full_path, 'config.yaml')
            shutil.copyfile(hypes_yaml, save_name)

    return full_path



def create_model(hypes):
    """
    Import the module "models/[model_name].py

    Parameters
    __________
    hypes : dict
        Dictionary containing parameters.

    Returns
    -------
    model : opencood,object
        Model object.
    """
    backbone_name = hypes['model']['core_method']
    backbone_config = hypes['model']['args']
    mode = hypes['model']['mode']

    model_filename = f"opencood.models.attack_modules.{mode}.{backbone_name}"
    model_lib = importlib.import_module(model_filename)
    model = None
    target_model_name = backbone_name.replace('_', '')

    for name, cls in model_lib.__dict__.items():
        if name.lower() == target_model_name.lower():
            model = cls

    if model is None:
        print('backbone not found in models folder. Please make sure you '
              'have a python file named %s and has a class '
              'called %s ignoring upper/lower case' % (model_filename,
                                                       target_model_name))
        exit(0)
    instance = model(backbone_config)
    return instance


def create_loss(hypes):
    """
    Create the loss function based on the given loss name.

    Parameters
    ----------
    hypes : dict
        Configuration params for training.
    Returns
    -------
    criterion : opencood.object
        The loss function.
    """
    loss_func_name = hypes['loss']['core_method']
    loss_func_config = hypes['loss']['args']

    loss_filename = "opencood.loss." + loss_func_name
    loss_lib = importlib.import_module(loss_filename)
    loss_func = None
    target_loss_name = loss_func_name.replace('_', '')

    for name, lfunc in loss_lib.__dict__.items():
        if name.lower() == target_loss_name.lower():
            loss_func = lfunc
    
    if loss_func is None:
        print('loss function not found in loss folder. Please make sure you '
              'have a python file named %s and has a class '
              'called %s ignoring upper/lower case' % (loss_filename,
                                                       target_loss_name))
        exit(0)

    criterion = loss_func(loss_func_config)
    return criterion


def setup_optimizer(hypes, model):
    """
    Create optimizer corresponding to the yaml file

    Parameters
    ----------
    hypes : dict
        The training configurations.
    model : opencood model
        The pytorch model
    """
    method_dict = hypes['optimizer']
    optimizer_method = getattr(optim, method_dict['core_method'], None)
    print('optimizer method is: %s' % optimizer_method)

    if not optimizer_method:
        raise ValueError('{} is not supported'.format(method_dict['name']))
    if 'args' in method_dict:
        return optimizer_method(filter(lambda p: p.requires_grad,
                                       model.parameters()),
                                lr=method_dict['lr'],
                                **method_dict['args'])
    else:
        return optimizer_method(filter(lambda p: p.requires_grad,
                                       model.parameters()),
                                lr=method_dict['lr'])


def setup_lr_schedular(hypes, optimizer, n_iter_per_epoch):
    """
    Set up the learning rate schedular.

    Parameters
    ----------
    hypes : dict
        The training configurations.

    optimizer : torch.optimizer
    """
    lr_schedule_config = hypes['lr_scheduler']

    if lr_schedule_config['core_method'] == 'step':
        from torch.optim.lr_scheduler import StepLR
        step_size = lr_schedule_config['step_size']
        gamma = lr_schedule_config['gamma']
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif lr_schedule_config['core_method'] == 'multistep':
        from torch.optim.lr_scheduler import MultiStepLR
        milestones = lr_schedule_config['step_size']
        gamma = lr_schedule_config['gamma']
        scheduler = MultiStepLR(optimizer,
                                milestones=milestones,
                                gamma=gamma)

    elif lr_schedule_config['core_method'] == 'exponential':
        print('ExponentialLR is chosen for lr scheduler')
        from torch.optim.lr_scheduler import ExponentialLR
        gamma = lr_schedule_config['gamma']
        scheduler = ExponentialLR(optimizer, gamma)

    elif lr_schedule_config['core_method'] == 'cosineannealwarm':
        print('cosine annealing is chosen for lr scheduler')
        from timm.scheduler.cosine_lr import CosineLRScheduler

        num_steps = lr_schedule_config['epoches'] * n_iter_per_epoch
        warmup_lr = lr_schedule_config['warmup_lr']
        warmup_steps = lr_schedule_config['warmup_epoches'] * n_iter_per_epoch
        lr_min = lr_schedule_config['lr_min']

        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min=lr_min,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )
    else:
        sys.exit('not supported lr schedular')

    return scheduler


def to_device(inputs, device):
    if isinstance(inputs, list):
        return [to_device(x, device) for x in inputs]
    elif isinstance(inputs, dict):
        return {k: to_device(v, device) for k, v in inputs.items()}
    else:
        if isinstance(inputs, int) or isinstance(inputs, float) \
                or isinstance(inputs, str):
            return inputs
        return inputs.to(device)
