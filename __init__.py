from .nodes import *

NODE_CLASS_MAPPINGS = {
    "Automatic CFG": simpleDynamicCFG,
    "Automatic CFG - Negative": simpleDynamicCFGlerpUncond,
    "Automatic CFG - Advanced settings": advancedDynamicCFG,
}
