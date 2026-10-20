"""
FaceForge AI - Compatibility & Runtime Patching Module
Applies standard torchvision and basicsr compatibility patches before any models are loaded.
"""

import sys
import logging

logger = logging.getLogger(__name__)


def apply_compat_patches():
    """Apply standard compatibility patches for torchvision and basicsr."""
    # 1. Patch torchvision.transforms.functional_tensor
    try:
        import torchvision.transforms.functional as F
        if "torchvision.transforms.functional_tensor" not in sys.modules:
            sys.modules["torchvision.transforms.functional_tensor"] = F
    except Exception as e:
        logger.debug("Failed to patch functional_tensor: %s", e)

    # 2. Patch basicsr Registry if basicsr is installed
    try:
        import basicsr.utils.registry as reg

        orig_register = getattr(reg.Registry, "_patched", False)
        if not orig_register:
            def safe_do_register(self, name, obj, suffix=None):
                if suffix:
                    name = f"{name}_{suffix}"
                self._obj_map[name] = obj

            def safe_register(self, obj=None, suffix=None, **kwargs):
                if obj is None:
                    def deco(func_or_class):
                        name = func_or_class.__name__
                        self._do_register(name, func_or_class, suffix=suffix)
                        return func_or_class
                    return deco
                name = obj.__name__
                self._do_register(name, obj, suffix=suffix)
                return obj

            reg.Registry._do_register = safe_do_register
            reg.Registry.register = safe_register
            reg.Registry._patched = True
            logger.debug("Applied basicsr Registry patch.")
    except Exception as e:
        logger.debug("basicsr Registry patch not applied: %s", e)


# Apply immediately upon import
apply_compat_patches()
