import os
from PIL import Image
import argparse
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.static import InputSpec
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from config import get_config
from config import update_config
from pit import build_pit as build_model
from datasets import get_val_transforms

import warnings
warnings.filterwarnings('ignore')

def get_arguments():
    """return argumeents, this will overwrite the config after loading yaml file"""
    parser = argparse.ArgumentParser('PiT')
    parser.add_argument('-cfg', type=str, default='configs/pit_ti.yaml')
    parser.add_argument('-dataset', type=str, default=None)
    parser.add_argument('-batch_size', type=int, default=None)
    parser.add_argument('-image_size', type=int, default=224)
    parser.add_argument('-data_path', type=str, default=None)
    parser.add_argument('-save_path', type=str, default=None)
    parser.add_argument('-ngpus', type=int, default=None)
    parser.add_argument('-resume', type=str, default=None)
    parser.add_argument('-last_epoch', type=int, default=None)
    parser.add_argument('-teacher_model', type=str, default=None)
    parser.add_argument('-eval', action='store_true')
    parser.add_argument('-amp', action='store_true')

    # Some configs you should change:
    parser.add_argument('-pretrained', type=str, default='output/Best_PiT')
    parser.add_argument('-img_path', type=str, default='images/ILSVRC2012_val_00004506.JPEG')
    arguments = parser.parse_args()
    return arguments

@paddle.no_grad()
def main(config):
    # define model
    model = build_model(config)
    model.eval()

    # load weights
    if (config.MODEL.PRETRAINED).endswith('.pdparams'):
        raise ValueError(f'{config.MODEL.PRETRAINED} should not contain .pdparams')
    assert os.path.isfile(config.MODEL.PRETRAINED + '.pdparams') is True
    model_state = paddle.load(config.MODEL.PRETRAINED+'.pdparams')
    model.set_dict(model_state)
    print(f"----- Pretrained: Load model state from {config.MODEL.PRETRAINED}")

    # decorate model with jit.save
    model = paddle.jit.to_static(
        model,
        input_spec=[
            InputSpec(
                shape=[None, 3, config.DATA.IMAGE_SIZE, config.DATA.IMAGE_SIZE], dtype='float32')
        ])
    # save inference model
    paddle.jit.save(model, os.path.join(config.SAVE, "inference"))
    print(f"inference model has been saved into {config.SAVE}")



if __name__ == "__main__":
    # config is updated by: (1) config.py, (2) yaml file, (3) arguments
    arguments = get_arguments()
    config = get_config()
    config = update_config(config, arguments)

    main(config)

    """
    This website could be used to check the validity of classification:
    https://blog.csdn.net/winycg/article/details/101722445
    """