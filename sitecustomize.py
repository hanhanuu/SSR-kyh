"""
Ensure distutils.version is available for torch.utils.tensorboard.
This avoids AttributeError when setuptools' distutils shim is used.
"""
try:
    import distutils.version  # noqa: F401
except Exception:
    pass
