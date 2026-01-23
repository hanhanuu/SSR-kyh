from .mmdet_train import custom_train_detector
from mmseg.apis import train_segmentor
from mmdet.apis import train_detector
import sys
import os

# 添加项目路径
sys.path.insert(0, '/home/hfut/SSR/SSR/mmdetection3d')
sys.path.insert(0, '/home/hfut/SSR/SSR/projects')

# 强制导入 SSR 插件，确保注册器被触发
def force_import_ssr():
    """Force import SSR to register the model."""
    try:
        # 先导入插件
        import projects.mmdet3d_plugin
        print("Successfully imported projects.mmdet3d_plugin")
        
        # 从 DETECTORS 注册表获取所有已注册的模型
        from mmdet.models.builder import DETECTORS
        print(f"Before importing SSR: {list(DETECTORS.module_dict.keys())}")
        
        # 直接导入 SSR 类，这会触发 @DETECTORS.register_module() 装饰器
        from projects.mmdet3d_plugin.SSR.SSR import SSR
        
        # 验证是否已注册
        if 'SSR' in DETECTORS.module_dict:
            print("SUCCESS: SSR is registered in DETECTORS registry")
        else:
            print("FAILED: SSR is not registered, trying manual registration")
            # 手动注册
            DETECTORS.register_module(name='SSR', module=SSR)
            print("Manually registered SSR")
            
        print(f"After importing SSR: {list(DETECTORS.module_dict.keys())}")
        
    except Exception as e:
        print(f"Error during SSR import: {e}")
        import traceback
        traceback.print_exc()
        raise

# 立即执行导入
force_import_ssr()

import argparse
import copy
import mmcv
import os
import time
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from os import path as osp

from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
#from mmdet3d.apis import train_model

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed
from mmseg import __version__ as mmseg_version

from mmcv.utils import TORCH_VERSION, digit_version

import cv2
cv2.setNumThreads(1)
def custom_train_model(model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                timestamp=None,
                eval_model=None,
                meta=None):
    """A function wrapper for launching model training according to cfg.

    Because we need different eval_hook in runner. Should be deprecated in the
    future.
    """
    if cfg.model.type in ['EncoderDecoder3D']:
        assert False
    else:
        custom_train_detector(
            model,
            dataset,
            cfg,
            distributed=distributed,
            validate=validate,
            timestamp=timestamp,
            eval_model=eval_model,
            meta=meta)


def train_model(model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                timestamp=None,
                meta=None):
    """A function wrapper for launching model training according to cfg.

    Because we need different eval_hook in runner. Should be deprecated in the
    future.
    """
    if cfg.model.type in ['EncoderDecoder3D']:
        train_segmentor(
            model,
            dataset,
            cfg,
            distributed=distributed,
            validate=validate,
            timestamp=timestamp,
            meta=meta)
    else:
        train_detector(
            model,
            dataset,
            cfg,
            distributed=distributed,
            validate=validate,
            timestamp=timestamp,
            meta=meta)
