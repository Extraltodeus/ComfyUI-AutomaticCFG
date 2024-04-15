from .nodes import *

NODE_CLASS_MAPPINGS = {
    "Automatic CFG": simpleDynamicCFG,
    "Automatic CFG - Negative": simpleDynamicCFGlerpUncond,
    "Automatic CFG - No uncond": simpleDynamicCFGNoUncond,
    "Automatic CFG - Advanced settings": advancedDynamicCFG,
}
