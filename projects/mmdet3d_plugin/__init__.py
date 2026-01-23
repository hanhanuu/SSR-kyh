from .core.bbox.assigners.hungarian_assigner_3d import HungarianAssigner3D
from .core.bbox.coders.nms_free_coder import NMSFreeCoder
from .core.bbox.match_costs import BBox3DL1Cost
from .core.evaluation.eval_hooks import CustomDistEvalHook
from .datasets.pipelines import (
  PhotoMetricDistortionMultiViewImage, PadMultiViewImage, 
  NormalizeMultiviewImage,  CustomCollect3D)
from .models.backbones.vovnet import VoVNet
from .models.utils import *
from .models.opt.adamw import AdamW2
from .SSR import *

# Strip dataset-only keys from registry builds that should never receive them.
def _strip_cfg_keys(node, keys):
    if isinstance(node, dict):
        return {k: _strip_cfg_keys(v, keys) for k, v in node.items() if k not in keys}
    if isinstance(node, list):
        return [_strip_cfg_keys(v, keys) for v in node]
    if isinstance(node, tuple):
        return tuple(_strip_cfg_keys(v, keys) for v in node)
    return node


def _patch_build_from_cfg():
    try:
        from mmcv.utils import registry as _mmcv_registry
        import mmcv.utils as _mmcv_utils
        import mmcv.cnn.builder as _mmcv_cnn_builder
    except Exception:
        return

    if getattr(_mmcv_registry.build_from_cfg, "_ssr_clean", False):
        return

    def _clean_build_from_cfg(cfg, registry, default_args=None):
        import inspect
        from mmcv.utils.registry import Registry as _Registry

        if not isinstance(cfg, dict):
            raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
        if 'type' not in cfg:
            if default_args is None or 'type' not in default_args:
                raise KeyError(
                    '`cfg` or `default_args` must contain the key "type", '
                    f'but got {cfg}\n{default_args}')
        if not isinstance(registry, _Registry):
            raise TypeError('registry must be an mmcv.Registry object, '
                            f'but got {type(registry)}')
        if not (isinstance(default_args, dict) or default_args is None):
            raise TypeError('default_args must be a dict or None, '
                            f'but got {type(default_args)}')

        args = cfg.copy()
        if default_args is not None:
            for name, value in default_args.items():
                args.setdefault(name, value)

        # Drop dataset-only keys for non-dataset registries.
        reg_name = getattr(registry, "name", None)
        if reg_name not in {'dataset', 'datasets', 'DATASETS'}:
            args.pop('ann_file', None)
            args.pop('map_ann_file', None)

        obj_type = args.pop('type')
        if isinstance(obj_type, str):
            obj_cls = registry.get(obj_type)
            if obj_cls is None:
                raise KeyError(
                    f'{obj_type} is not in the {registry.name} registry')
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(
                f'type must be a str or valid type, but got {type(obj_type)}')
        try:
            return obj_cls(**args)
        except Exception as e:
            raise type(e)(f'{obj_cls.__name__}: {e}')

    _clean_build_from_cfg._ssr_clean = True
    _mmcv_registry.build_from_cfg = _clean_build_from_cfg
    _mmcv_utils.build_from_cfg = _clean_build_from_cfg
    _mmcv_cnn_builder.build_from_cfg = _clean_build_from_cfg


_patch_build_from_cfg()


def _patch_match_cost_builder():
    try:
        from mmdet.core.bbox.match_costs import builder as _mc_builder
    except Exception:
        return

    if getattr(_mc_builder.build_match_cost, "_ssr_patched", False):
        return

    _orig = _mc_builder.build_match_cost

    def _wrapped(cfg, default_args=None):
        cfg = _strip_cfg_keys(cfg, {'ann_file', 'map_ann_file'})
        if default_args is not None:
            default_args = _strip_cfg_keys(default_args, {'ann_file', 'map_ann_file'})
        return _orig(cfg, default_args)

    _wrapped._ssr_patched = True
    _mc_builder.build_match_cost = _wrapped


_patch_match_cost_builder()


def _patch_match_cost_inits():
    try:
        import inspect
        from mmdet.core.bbox.match_costs import match_cost as _mc
    except Exception:
        return

    def _wrap_init(cls):
        orig = cls.__init__
        if getattr(orig, "_ssr_patched", False):
            return
        sig = inspect.signature(orig)
        has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        valid = {p.name for p in sig.parameters.values() if p.name != 'self'}

        def _new(self, *args, **kwargs):
            kwargs.pop('ann_file', None)
            kwargs.pop('map_ann_file', None)
            if not has_varkw:
                kwargs = {k: v for k, v in kwargs.items() if k in valid}
            return orig(self, *args, **kwargs)

        _new._ssr_patched = True
        cls.__init__ = _new

    for name in dir(_mc):
        obj = getattr(_mc, name)
        if isinstance(obj, type) and hasattr(obj, "__init__"):
            _wrap_init(obj)


_patch_match_cost_inits()
