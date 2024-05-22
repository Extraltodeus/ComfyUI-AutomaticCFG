import torch
from torch import nn, einsum
from einops import rearrange, repeat

def exists(val):
    return val is not None

# taken from comfy.ldm.modules
class temperature_patcher():
    def __init__(self, temperature):
        self.temperature = temperature
    
    def attention_basic_with_temperature(self, q, k, v, extra_options, mask=None, attn_precision=None):
        heads = extra_options['n_heads']

        b, _, dim_head = q.shape
        dim_head //= heads
        scale = dim_head ** -0.5

        h = heads
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, -1, heads, dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * heads, -1, dim_head)
            .contiguous(),
            (q, k, v),
        )

        # force cast to fp32 to avoid overflowing
        if attn_precision == torch.float32:
            sim = einsum('b i d, b j d -> b i j', q.float(), k.float()) * scale
        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * scale

        del q, k

        if exists(mask):
            if mask.dtype == torch.bool:
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                sim.masked_fill_(~mask, max_neg_value)
            else:
                if len(mask.shape) == 2:
                    bs = 1
                else:
                    bs = mask.shape[0]
                mask = mask.reshape(bs, -1, mask.shape[-2], mask.shape[-1]).expand(b, heads, -1, -1).reshape(-1, mask.shape[-2], mask.shape[-1])
                sim.add_(mask)
        
        # attention, what we cannot get enough of
        sim = sim.div(self.temperature).softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', sim.to(v.dtype), v)
        out = (
            out.unsqueeze(0)
            .reshape(b, heads, -1, dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, -1, heads * dim_head)
        )
        return out

layers_SD15 = {
    "input":[1,2,4,5,7,8],
    "middle":[0],
    "output":[3,4,5,6,7,8,9,10,11],
}

layers_SDXL = {
    "input":[4,5,7,8],
    "middle":[0],
    "output":[0,1,2,3,4,5],
}

class ExperimentalTemperaturePatch:
    @classmethod
    def INPUT_TYPES(s):
        required_inputs = {f"{key}_{layer}": ("BOOLEAN", {"default": False}) for key, layers in s.TOGGLES.items() for layer in layers}
        required_inputs["model"] = ("MODEL",)
        required_inputs["Temperature"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1, "round": 0.01})
        return {"required": required_inputs}
    
    TOGGLES = {}
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("Model",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Automatic_CFG"

    def patch(self, model, Temperature, **kwargs):
        m = model.clone()
        for key, toggle_enabled in kwargs.items():
            if key.split("_")[0] in ["input","middle","output"] and toggle_enabled:
                patcher = temperature_patcher(Temperature)
                m.set_model_attn1_replace(patcher.attention_basic_with_temperature, key.split("_")[0], int(key.split("_")[1]))
        return (m, )
    
ExperimentalTemperaturePatchSDXL = type("ExperimentalTemperaturePatch_SDXL", (ExperimentalTemperaturePatch,), {"TOGGLES": layers_SDXL})
ExperimentalTemperaturePatchSD15 = type("ExperimentalTemperaturePatch_SD15", (ExperimentalTemperaturePatch,), {"TOGGLES": layers_SD15})