from .nodes import *
from .nodes_sag_custom import *

NODE_CLASS_MAPPINGS = {
    "Automatic CFG": simpleDynamicCFG,
    "Automatic CFG - Negative": simpleDynamicCFGlerpUncond,
    "Automatic CFG - Warp Drive": simpleDynamicCFGwarpDrive,
    "Automatic CFG - Advanced": advancedDynamicCFG,
    "Automatic CFG - Post rescale only": postCFGrescaleOnly,
    "Automatic CFG - Unpatch function": simpleDynamicCFGunpatch,
    "SAG delayed activation": SelfAttentionGuidanceCustom,
}
