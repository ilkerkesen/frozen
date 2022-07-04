import os
import os.path as osp

import pytorch_lightning as pl

def create_callbacks(config, log_dir):
    checkpoints_path = osp.join(log_dir, 'checkpoints')
    config['checkpoint']['dirpath'] = checkpoints_path
    checkpoint_callback = pl.callbacks.ModelCheckpoint(**config['checkpoint'])
    last_ckpt = osp.join(checkpoints_path, 'last.ckpt')
    last_ckpt = last_ckpt if osp.isfile(last_ckpt) else None
    ckpt_path = config['trainer']['resume_from_checkpoint']

    if last_ckpt is not None and ckpt_path is not None:
        raise Exception('resume checkpoint passed (last.ckpt exists already)')

    ckpt_path = last_ckpt if ckpt_path is None else ckpt_path
    if ckpt_path is not None and not osp.isfile(ckpt_path):
        raise Exception('ckpt does not exist at {}'.format(ckpt_path))

    return [checkpoint_callback], ckpt_path


def create_logger(config):
    assert config['logger'].get('version') is not None
    if config['logger']['version'] == 'debug':
        return None
    config['logger']['save_dir'] = osp.abspath(
        osp.expanduser(config['logger']['save_dir']))
    if config['logger']['name'] is None:
        architecture = config['model']['name']
        config['logger']['name'] = f'{architecture}'
    logger = pl.loggers.TensorBoardLogger(**config['logger'])
    return logger


def process_config(config):
    model_config = config.get('model', {})
    dataset_config = config.get('dataset', {})
    N = model_config.get('num_image_tokens', 2)
    dataset_config['num_image_tokens'] = N
    config['dataset'] = dataset_config
    return config