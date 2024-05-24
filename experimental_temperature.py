import torch
from torch import nn, einsum
from einops import rearrange, repeat
import torch.nn.functional as F
import math
from comfy import model_management
import types
import os

def exists(val):
    return val is not None

# better than a division by 0 hey
abs_mean = lambda x: torch.where(torch.isnan(x) | torch.isinf(x), torch.zeros_like(x), x).abs().mean()

class temperature_patcher():
    def __init__(self, temperature, layer_name="None"):
        self.temperature = temperature
        self.layer_name  = layer_name
    
    # taken from comfy.ldm.modules
    def attention_basic_with_temperature(self, q, k, v, extra_options, mask=None, attn_precision=None):
        if isinstance(extra_options, int):
            heads = extra_options
        else:
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
        sim = sim.div(self.temperature if self.temperature > 0 else abs_mean(sim)).softmax(dim=-1)

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
        required_inputs["Temperature"]  = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "round": 0.01})
        required_inputs["Attention"]    = (["both","self","cross"],)
        return {"required": required_inputs}
    
    TOGGLES = {}
    RETURN_TYPES = ("MODEL","STRING",)
    RETURN_NAMES = ("Model","String",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Automatic_CFG/Standalone_temperature_patches"

    def patch(self, model, Temperature, Attention, **kwargs):
        m = model.clone()
        levels = ["input","middle","output"]
        parameters_output = {level:[] for level in levels}
        for key, toggle_enabled in kwargs.items():
            current_level = key.split("_")[0]
            if current_level in levels and toggle_enabled:
                b_number = int(key.split("_")[1])
                parameters_output[current_level].append(b_number)
                patcher = temperature_patcher(Temperature,key)

                if Attention in ["both","self"]:
                    m.set_model_attn1_replace(patcher.attention_basic_with_temperature, current_level, b_number)
                if Attention in ["both","cross"]:
                    m.set_model_attn2_replace(patcher.attention_basic_with_temperature, current_level, b_number)

        parameters_as_string = "\n".join(f"{k}: {','.join(map(str, v))}" for k, v in parameters_output.items())
        parameters_as_string = f"Temperature: {Temperature}\n{parameters_as_string}\nAttention: {Attention}"
        return (m, parameters_as_string,)
    
ExperimentalTemperaturePatchSDXL = type("ExperimentalTemperaturePatch_SDXL", (ExperimentalTemperaturePatch,), {"TOGGLES": layers_SDXL})
ExperimentalTemperaturePatchSD15 = type("ExperimentalTemperaturePatch_SD15", (ExperimentalTemperaturePatch,), {"TOGGLES": layers_SD15})

class CLIPTemperaturePatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": { "clip": ("CLIP",),
                              "Temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              }}
    
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "patch"
    CATEGORY = "model_patches/Automatic_CFG/Standalone_temperature_patches"
    
    def patch(self, clip, Temperature):
        def custom_optimized_attention(device, mask=None, small_input=True):
            return temperature_patcher(Temperature).attention_basic_with_temperature
        
        def new_forward(self, x, mask=None, intermediate_output=None):
            optimized_attention = custom_optimized_attention(x.device, mask=mask is not None, small_input=True)

            if intermediate_output is not None:
                if intermediate_output < 0:
                    intermediate_output = len(self.layers) + intermediate_output

            intermediate = None
            for i, l in enumerate(self.layers):
                x = l(x, mask, optimized_attention)
                if i == intermediate_output:
                    intermediate = x.clone()
            return x, intermediate

        m = clip.clone()

        clip_encoder_instance = m.cond_stage_model.clip_l.transformer.text_model.encoder
        clip_encoder_instance.forward = types.MethodType(new_forward, clip_encoder_instance)

        if getattr(m.cond_stage_model, f"clip_g", None) is not None:
            clip_encoder_instance_g = m.cond_stage_model.clip_g.transformer.text_model.encoder
            clip_encoder_instance_g.forward = types.MethodType(new_forward, clip_encoder_instance_g)
        
        return (m,)

class CLIPTemperaturePatchDual:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": { "clip": ("CLIP",),
                              "Temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                              "CLIP_Model": (["clip_g","clip_l","both"],),
                              }}
    
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "patch"
    CATEGORY = "model_patches/Automatic_CFG/Standalone_temperature_patches"
    
    def patch(self, clip, Temperature, CLIP_Model):
        def custom_optimized_attention(device, mask=None, small_input=True):
            return temperature_patcher(Temperature, "CLIP").attention_basic_with_temperature
        
        def new_forward(self, x, mask=None, intermediate_output=None):
            optimized_attention = custom_optimized_attention(x.device, mask=mask is not None, small_input=True)

            if intermediate_output is not None:
                if intermediate_output < 0:
                    intermediate_output = len(self.layers) + intermediate_output

            intermediate = None
            for i, l in enumerate(self.layers):
                x = l(x, mask, optimized_attention)
                if i == intermediate_output:
                    intermediate = x.clone()
            return x, intermediate

        m = clip.clone()

        if CLIP_Model in ["clip_l","both"]:
            clip_encoder_instance = m.cond_stage_model.clip_l.transformer.text_model.encoder
            clip_encoder_instance.forward = types.MethodType(new_forward, clip_encoder_instance)

        if CLIP_Model in ["clip_g","both"]:
            if getattr(m.cond_stage_model, f"clip_g", None) is not None:
                clip_encoder_instance_g = m.cond_stage_model.clip_g.transformer.text_model.encoder
                clip_encoder_instance_g.forward = types.MethodType(new_forward, clip_encoder_instance_g)
        
        return (m,)
