from .nodes import *
from .nodes_sag_custom import *

NODE_CLASS_MAPPINGS = {
    "Automatic CFG": simpleDynamicCFG,
    "Automatic CFG - Negative": simpleDynamicCFGlerpUncond,
    "Automatic CFG - Fastest": simpleDynamicCFGHighSpeed,
    "Automatic CFG - Advanced": advancedDynamicCFG,
    "Automatic CFG - Post rescale only": postCFGrescaleOnly,
    "SAG delayed activation": SelfAttentionGuidanceCustom,
}
