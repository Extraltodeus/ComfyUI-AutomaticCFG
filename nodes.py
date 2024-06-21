import math
from copy import deepcopy
from torch.nn import Upsample
import comfy.model_management as model_management
from comfy.model_patcher import set_model_options_patch_replace
from comfy.ldm.modules.attention import attention_basic, attention_xformers, attention_pytorch, attention_split, attention_sub_quad, optimized_attention_for_device
from .experimental_temperature import temperature_patcher
import comfy.samplers
import comfy.utils
import numpy as np
import torch
import torch.nn.functional as F
from colorama import Fore, Style
import json
import os
import random
import base64

original_sampling_function = None
current_dir = os.path.dirname(os.path.realpath(__file__))
json_preset_path = os.path.join(current_dir, 'presets')
attnfunc = optimized_attention_for_device(model_management.get_torch_device())
check_string   = "UEFUUkVPTi50eHQ="
support_string = b'CgoKClRoYW5rIHlvdSBmb3IgdXNpbmcgbXkgbm9kZXMhCgpJZiB5b3UgZW5qb3kgaXQsIHBsZWFzZSBjb25zaWRlciBzdXBwb3J0aW5nIG1lIG9uIFBhdHJlb24gdG8ga2VlcCB0aGUgbWFnaWMgZ29pbmchCgpWaXNpdDoKCmh0dHBzOi8vd3d3LnBhdHJlb24uY29tL2V4dHJhbHRvZGV1cwoKCgo='

def support_function():
    if base64.b64decode(check_string).decode('utf8') not in os.listdir(current_dir):
        print(base64.b64decode(check_string).decode('utf8'))
        print(base64.b64decode(support_string).decode('utf8'))

def sampling_function_patched(model, x, timestep, uncond, cond, cond_scale, model_options={}, seed=None, **kwargs):

    cond_copy   = cond
    uncond_copy = uncond

    for fn in model_options.get("sampler_patch_model_pre_cfg_function", []):
        args = {"model": model, "sigma": timestep, "model_options": model_options}
        model, model_options = fn(args)

    if "sampler_pre_cfg_function" in model_options:
        uncond, cond, cond_scale = model_options["sampler_pre_cfg_function"](
            sigma=timestep, uncond=uncond, cond=cond, cond_scale=cond_scale
        )
        
    if math.isclose(cond_scale, 1.0) and model_options.get("disable_cfg1_optimization", False) == False:
        uncond_ = None
    else:
        uncond_ = uncond

    conds = [cond, uncond_]

    out = comfy.samplers.calc_cond_batch(model, conds, x, timestep, model_options)
    cond_pred = out[0]
    uncond_pred = out[1]

    if "sampler_cfg_function" in model_options:
        args = {"cond": x - cond_pred, "uncond": x - uncond_pred, "cond_scale": cond_scale, "timestep": timestep, "input": x, "sigma": timestep,
                "cond_denoised": cond_pred, "uncond_denoised": uncond_pred, "model": model, "model_options": model_options, "cond_pos": cond_copy, "cond_neg": uncond_copy}
        cfg_result = x - model_options["sampler_cfg_function"](args)
    else:
        cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale

    for fn in model_options.get("sampler_post_cfg_function", []):
        args = {"denoised": cfg_result, "cond": cond_copy, "uncond": uncond_copy, "model": model, "uncond_denoised": uncond_pred, "cond_denoised": cond_pred,
                "sigma": timestep, "model_options": model_options, "input": x}
        cfg_result = fn(args)
        
    return cfg_result

def monkey_patching_comfy_sampling_function():
    global original_sampling_function

    if original_sampling_function is None:
        original_sampling_function = comfy.samplers.sampling_function
    # Make sure to only patch once
    if hasattr(comfy.samplers.sampling_function, '_automatic_cfg_decorated'):
        return
    comfy.samplers.sampling_function = sampling_function_patched
    comfy.samplers.sampling_function._automatic_cfg_decorated = True # flag to check monkey patch

def make_sampler_pre_cfg_function(minimum_sigma_to_disable_uncond=0, maximum_sigma_to_enable_uncond=1000000, disabled_cond_start=10000,disabled_cond_end=10000):
    def sampler_pre_cfg_function(sigma, uncond, cond, cond_scale, **kwargs):
        if sigma[0] < minimum_sigma_to_disable_uncond or sigma[0] > maximum_sigma_to_enable_uncond:
            uncond = None
        if sigma[0] <= disabled_cond_start and sigma[0] > disabled_cond_end:
            cond = None
        return uncond, cond, cond_scale
    return sampler_pre_cfg_function

def get_entropy(tensor):
    hist = np.histogram(tensor.cpu(), bins=100)[0]
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))

def map_sigma(sigma, sigmax, sigmin):
    return 1 + ((sigma - sigmax) * (0 - 1)) / (sigmin - sigmax)

def center_latent_mean_values(latent, per_channel, mult):
    for b in range(len(latent)):
        if per_channel:
            for c in range(len(latent[b])):
                latent[b][c] -= latent[b][c].mean() * mult
        else:
            latent[b] -= latent[b].mean() * mult
    return latent

def get_denoised_ranges(latent, measure="hard", top_k=0.25):
    chans = []
    for x in range(len(latent)):
        max_values = torch.topk(latent[x] - latent[x].mean() if measure == "range" else latent[x], k=int(len(latent[x])*top_k), largest=True).values
        min_values = torch.topk(latent[x] - latent[x].mean() if measure == "range" else latent[x], k=int(len(latent[x])*top_k), largest=False).values
        max_val = torch.mean(max_values).item()
        min_val = abs(torch.mean(min_values).item()) if measure == "soft" else torch.mean(torch.abs(min_values)).item()
        denoised_range = (max_val + min_val) / 2
        chans.append(denoised_range**2 if measure == "hard_squared" else denoised_range)
    return chans

def get_sigmin_sigmax(model):
    model_sampling = model.model.model_sampling
    sigmin = model_sampling.sigma(model_sampling.timestep(model_sampling.sigma_min))
    sigmax = model_sampling.sigma(model_sampling.timestep(model_sampling.sigma_max))
    return sigmin, sigmax

def gaussian_similarity(x, y, sigma=1.0):
    diff = (x - y) ** 2
    return torch.exp(-diff / (2 * sigma ** 2))
    
def check_skip(sigma, high_sigma_threshold, low_sigma_threshold):
    return sigma > high_sigma_threshold or sigma < low_sigma_threshold

def max_abs(tensors):
    shape = tensors.shape
    tensors = tensors.reshape(shape[0], -1)
    tensors_abs = torch.abs(tensors)
    max_abs_idx = torch.argmax(tensors_abs, dim=0)
    result = tensors[max_abs_idx, torch.arange(tensors.shape[1])]
    return result.reshape(shape[1:])

def gaussian_kernel(size: int, sigma: float):
    x = torch.arange(size) - size // 2
    gauss = torch.exp(-x**2 / (2 * sigma**2))
    kernel = gauss / gauss.sum()
    return kernel.view(1, size) * kernel.view(size, 1)

def blur_tensor(tensor, kernel_size = 9, sigma = 2.0):
    tensor = tensor.unsqueeze(0)
    C = tensor.size(1)
    kernel = gaussian_kernel(kernel_size, sigma)
    kernel = kernel.expand(C, 1, kernel_size, kernel_size).to(tensor.device).to(dtype=tensor.dtype, device=tensor.device)
    padding = kernel_size // 2
    tensor = F.pad(tensor, (padding, padding, padding, padding), mode='reflect')
    blurred_tensor = F.conv2d(tensor, kernel, groups=C)
    return blurred_tensor.squeeze(0)

def smallest_distances(tensors):
    if all(torch.equal(tensors[0], tensor) for tensor in tensors[1:]):
        return tensors[0]
    set_device = tensors.device
    min_val = torch.full(tensors[0].shape, float("inf")).to(set_device)
    result  = torch.zeros_like(tensors[0])
    for idx1, t1 in enumerate(tensors):
        temp_diffs = torch.zeros_like(tensors[0])
        for idx2, t2 in enumerate(tensors):
            if idx1 != idx2:
                temp_diffs += torch.abs(torch.sub(t1, t2))
        min_val = torch.minimum(min_val, temp_diffs)
        mask    = torch.eq(min_val,temp_diffs)
        result[mask] = t1[mask]
    return result

def rescale(tensor, multiplier=2):
    batch, seq_length, features = tensor.shape
    H = W = int(seq_length**0.5)
    tensor_reshaped = tensor.view(batch, features, H, W)
    new_H = new_W = int(H * multiplier)
    resized_tensor = F.interpolate(tensor_reshaped, size=(new_H, new_W), mode='bilinear', align_corners=False)
    return resized_tensor.view(batch, new_H * new_W, features)

# from  https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475
def slerp(high, low, val):
    dims = low.shape

    #flatten to batches
    low = low.reshape(dims[0], -1)
    high = high.reshape(dims[0], -1)

    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)

    # in case we divide by zero
    low_norm[low_norm != low_norm] = 0.0
    high_norm[high_norm != high_norm] = 0.0

    omega = torch.acos((low_norm*high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res.reshape(dims)

normalize_tensor = lambda x: x / x.norm()

def random_swap(tensors, proportion=1):
    num_tensors = tensors.shape[0]
    if num_tensors < 2: return tensors[0],0
    tensor_size = tensors[0].numel()
    if tensor_size < 100: return tensors[0],0

    true_count = int(tensor_size * proportion)
    mask = torch.cat((torch.ones(true_count, dtype=torch.bool, device=tensors[0].device), 
                      torch.zeros(tensor_size - true_count, dtype=torch.bool, device=tensors[0].device)))
    mask = mask[torch.randperm(tensor_size)].reshape(tensors[0].shape)
    if num_tensors == 2 and proportion < 1:
        index_tensor = torch.ones_like(tensors[0], dtype=torch.int64, device=tensors[0].device)
    else:
        index_tensor = torch.randint(1 if proportion < 1 else 0, num_tensors, tensors[0].shape, device=tensors[0].device)
    for i, t in enumerate(tensors):
        if i == 0: continue
        merge_mask = index_tensor == i & mask
        tensors[0][merge_mask] = t[merge_mask]
    return tensors[0]

def multi_tensor_check_mix(tensors):
    if tensors[0].numel() < 2 or len(tensors) < 2:
        return tensors[0]
    ref_tensor_shape = tensors[0].shape
    sequence_tensor = torch.arange(tensors[0].numel(), device=tensors[0].device) % len(tensors)
    reshaped_sequence = sequence_tensor.view(ref_tensor_shape)
    for i in range(len(tensors)):
        if i == 0: continue
        mask = reshaped_sequence == i
        tensors[0][mask] = tensors[i][mask]
    return tensors[0]

def sspow(input_tensor, p=2):
    return input_tensor.abs().pow(p) * input_tensor.sign()

def sspown(input_tensor, p=2):
    abs_t = input_tensor.abs()
    abs_t = (abs_t - abs_t.min()) / (abs_t.max() - abs_t.min())
    return abs_t.pow(p) * input_tensor.sign()

def gradient_merge(tensor1, tensor2, start_value=0, dim=0):
    if torch.numel(tensor1) <= 1: return tensor1
    if dim >= tensor1.dim(): dim = 0
    size = tensor1.size(dim)
    alpha = torch.linspace(start_value, 1-start_value, steps=size, device=tensor1.device).view([-1 if i == dim else 1 for i in range(tensor1.dim())])
    return tensor1 * alpha + tensor2 * (1 - alpha)

def save_tensor(input_tensor,name):
    if "rndnum" in name:
        rndnum = str(random.randint(100000,999999))
        name = name.replace("rndnum", rndnum)
    output_directory = os.path.join(current_dir, 'saved_tensors')
    os.makedirs(output_directory, exist_ok=True)
    output_file_path = os.path.join(output_directory, f"{name}.pt")
    torch.save(input_tensor, output_file_path)
    return input_tensor

def print_and_return(input_tensor, *args):
    for what_to_print in args:
        print(" ",what_to_print)
    return input_tensor

# Experimental testings
def normal_attention(q, k, v, mask=None):
    attention_scores = torch.matmul(q, k.transpose(-2, -1))
    d_k = k.size(-1)
    attention_scores = attention_scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    if mask is not None:
        attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
    attention_weights = F.softmax(attention_scores, dim=-1)
    output = torch.matmul(attention_weights, v)
    return output

def split_heads(x, n_heads):
    batch_size, seq_length, hidden_dim = x.size()
    head_dim = hidden_dim // n_heads
    x = x.view(batch_size, seq_length, n_heads, head_dim)
    return x.permute(0, 2, 1, 3)

def combine_heads(x, n_heads):
    batch_size, n_heads, seq_length, head_dim = x.size()
    hidden_dim = n_heads * head_dim
    x = x.permute(0, 2, 1, 3).contiguous()
    return x.view(batch_size, seq_length, hidden_dim)

def sparsemax(logits):
    logits_sorted, _ = torch.sort(logits, descending=True, dim=-1)
    cumulative_sum = torch.cumsum(logits_sorted, dim=-1) - 1
    rho = (logits_sorted > cumulative_sum / (torch.arange(logits.size(-1)) + 1).to(logits.device)).float()
    tau = (cumulative_sum / rho.sum(dim=-1, keepdim=True)).gather(dim=-1, index=rho.sum(dim=-1, keepdim=True).long() - 1)
    return torch.max(torch.zeros_like(logits), logits - tau)

def attnfunc_custom(q, k, v, n_heads, eval_string = ""):
    q = split_heads(q, n_heads)
    k = split_heads(k, n_heads)
    v = split_heads(v, n_heads)
    
    d_k = q.size(-1)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if eval_string == "":
        attn_weights = F.softmax(scores, dim=-1)
    else:
        attn_weights = eval(eval_string)

    output = torch.matmul(attn_weights, v)
    output = combine_heads(output, n_heads)
    return output

def min_max_norm(t):
    return (t - t.min()) / (t.max() - t.min())

class attention_modifier():
    def __init__(self, self_attn_mod_eval, conds = None):
        self.self_attn_mod_eval = self_attn_mod_eval
        self.conds = conds

    def modified_attention(self, q, k, v, extra_options, mask=None):
        
        """extra_options contains: {'cond_or_uncond': [1, 0], 'sigmas': tensor([14.6146], device='cuda:0'),
         'original_shape': [2, 4, 128, 128], 'transformer_index': 4, 'block': ('middle', 0),
         'block_index': 3, 'n_heads': 20, 'dim_head': 64, 'attn_precision': None}"""
        
        if "attnbc" in self.self_attn_mod_eval:
            attnbc = attention_basic(q, k, v, extra_options['n_heads'], mask)
        if "normattn" in self.self_attn_mod_eval:
            normattn = normal_attention(q, k, v, mask)
        if "attnxf" in self.self_attn_mod_eval:
            attnxf = attention_xformers(q, k, v, extra_options['n_heads'], mask)
        if "attnpy" in self.self_attn_mod_eval:
            attnpy = attention_pytorch(q, k, v, extra_options['n_heads'], mask)
        if "attnsp" in self.self_attn_mod_eval:
            attnsp = attention_split(q, k, v, extra_options['n_heads'], mask)
        if "attnsq" in self.self_attn_mod_eval:
            attnsq = attention_sub_quad(q, k, v, extra_options['n_heads'], mask)
        if "attnopt" in self.self_attn_mod_eval:
            attnopt = attnfunc(q, k, v, extra_options['n_heads'], mask)
        n_heads = extra_options['n_heads']
        if self.conds is not None:
            cond_pos_l = self.conds[0][..., :768].cuda()
            cond_neg_l = self.conds[1][..., :768].cuda()
            if self.conds[0].shape[-1] > 768:
                cond_pos_g = self.conds[0][..., 768:2048].cuda()
                cond_neg_g = self.conds[1][..., 768:2048].cuda()
        return eval(self.self_attn_mod_eval)

def experimental_functions(cond_input, method, exp_value, exp_normalize, pcp, psi, sigma, sigmax, attention_modifiers_input, args, model_options_copy, eval_string = ""):
    """
    There may or may not be an actual reasoning behind each of these methods.
    Some like the sine value have interesting properties. Enabled for both cond and uncond preds it somehow make them stronger.
    Note that there is a "normalize" toggle and it may change greatly the end result since some operation will totaly butcher the values.
    "theDaRkNeSs" for example without normalizing seems to darken if used for cond/uncond (not with the cond as the uncond or something).
    Maybe just with the positive. I don't remember. I leave it for now if you want to play around.

    The eval_string can be used to create the uncond replacement.
    I made it so it's split by semicolons and only the last split is the value in used.
    What is before is added in an array named "v".
    pcp is previous cond_pred
    psi is previous sigma
    args is the CFG function input arguments with the added cond/unconds (like the actual activation conditionings) named respectively "cond_pos" and "cond_neg"

    So if you write:

    pcp if sigma < 7 else -pcp;
    print("it works too just don't use the output I guess");
    v[0] if sigma < 14 else torch.zeros_like(cond);
    v[-1]*2
    
    Well the first line becomes v[0], second v[1] etc.
    The last one becomes the result.
    Note that it's just an example, I don't see much interest in that one.

    Using comfy.samplers.calc_cond_batch(args["model"], [args["cond_pos"], None], args["input"], args["timestep"], args["model_options"])[0]
    can work too.

    This whole mess has for initial goal to attempt to find the best way (or have some bruteforcing fun) to replace the uncond pred for as much as possible.
    Or simply to try things around :)
    """
    if method == "cond_pred":
        return cond_input
    default_device = cond_input.device
    # print()
    # print(get_entropy(cond))
    cond = cond_input.clone()
    cond_norm = cond.norm()
    if method == "amplify":
        mask = torch.abs(cond) >= 1
        cond_copy = cond.clone()
        cond = torch.pow(torch.abs(cond), ( 1 / exp_value)) * cond.sign()
        cond[mask] = torch.pow(torch.abs(cond_copy[mask]), exp_value) * cond[mask].sign()
    elif method == "root":
        cond = torch.pow(torch.abs(cond), ( 1 / exp_value)) * cond.sign()
    elif method == "power":
        cond = torch.pow(torch.abs(cond), exp_value) * cond.sign()
    elif method == "erf":
        cond = torch.erf(cond)
    elif method == "exp_erf":
        cond = torch.pow(torch.erf(cond), exp_value)
    elif method == "root_erf":
        cond = torch.erf(cond)
        cond = torch.pow(torch.abs(cond), 1 / exp_value ) * cond.sign()
    elif method == "erf_amplify":
        cond = torch.erf(cond)
        mask = torch.abs(cond) >= 1
        cond_copy = cond.clone()
        cond = torch.pow(torch.abs(cond), 1 / exp_value ) * cond.sign()
        cond[mask] = torch.pow(torch.abs(cond_copy[mask]), exp_value) * cond[mask].sign()
    elif method == "sine":
        cond = torch.sin(torch.abs(cond)) * cond.sign()
    elif method == "sine_exp":
        cond = torch.sin(torch.abs(cond)) * cond.sign()
        cond = torch.pow(torch.abs(cond), exp_value) * cond.sign()
    elif method == "sine_exp_diff":
        cond = torch.sin(torch.abs(cond)) * cond.sign()
        cond = torch.pow(torch.abs(cond_input), exp_value) * cond.sign() - cond
    elif method == "sine_exp_diff_to_sine":
        cond = torch.sin(torch.abs(cond)) * cond.sign()
        cond = torch.pow(torch.abs(cond), exp_value) * cond.sign() - cond
    elif method == "sine_root":
        cond = torch.sin(torch.abs(cond)) * cond.sign()
        cond = torch.pow(torch.abs(cond), ( 1 / exp_value)) * cond.sign()
    elif method == "sine_root_diff":
        cond = torch.sin(torch.abs(cond)) * cond.sign()
        cond = torch.pow(torch.abs(cond_input), 1 / exp_value) * cond.sign() - cond
    elif method == "sine_root_diff_to_sine":
        cond = torch.sin(torch.abs(cond)) * cond.sign()
        cond = torch.pow(torch.abs(cond), 1 / exp_value) * cond.sign() - cond
    elif method == "theDaRkNeSs":
        cond = torch.sin(cond)
        cond = torch.pow(torch.abs(cond), 1 / exp_value) * cond.sign() - cond
    elif method == "cosine":
        cond = torch.cos(torch.abs(cond)) * cond.sign()
    elif method == "sign":
        cond = cond.sign()
    elif method == "zero":
        cond = torch.zeros_like(cond)
    elif method in ["attention_modifiers_input_using_cond","attention_modifiers_input_using_uncond","subtract_attention_modifiers_input_using_cond","subtract_attention_modifiers_input_using_uncond"]:
        cond_to_use = args["cond_pos"] if method in ["attention_modifiers_input_using_cond","subtract_attention_modifiers_input_using_cond"] else args["cond_neg"]
        tmp_model_options = deepcopy(model_options_copy)
        for atm in attention_modifiers_input:
            if sigma <= atm['sigma_start'] and sigma > atm['sigma_end']:
                block_layers = {"input": atm['unet_block_id_input'], "middle": atm['unet_block_id_middle'], "output": atm['unet_block_id_output']}
                for unet_block in block_layers:
                    for unet_block_id in block_layers[unet_block].split(","):
                        if unet_block_id != "":
                            unet_block_id = int(unet_block_id)
                            tmp_model_options = set_model_options_patch_replace(tmp_model_options, attention_modifier(atm['self_attn_mod_eval'], [args["cond_pos"][0]["cross_attn"], args["cond_neg"][0]["cross_attn"]]if "cond" in atm['self_attn_mod_eval'] else None).modified_attention, atm['unet_attn'], unet_block, unet_block_id)
        
        cond = comfy.samplers.calc_cond_batch(args["model"], [cond_to_use], args["input"], args["timestep"], tmp_model_options)[0]
        if method in ["subtract_attention_modifiers_input_using_cond","subtract_attention_modifiers_input_using_uncond"]:
            cond = cond_input + (cond_input - cond) * exp_value

    elif method == "previous_average":
        if sigma > (sigmax - 1):
            cond = torch.zeros_like(cond)
        else:
            cond = (pcp / psi * sigma + cond) / 2
    elif method == "eval":
        if "condmix" in eval_string:
            def condmix(args, mult=2):
                cond_pos_tmp = deepcopy(args["cond_pos"])
                cond_pos_tmp[0]["cross_attn"] += (args["cond_pos"][0]["cross_attn"] - args["cond_neg"][0]["cross_attn"]*-1) * mult
                return cond_pos_tmp
        v = []
        evals_strings = eval_string.split(";")
        if len(evals_strings) > 1:
            for i in range(len(evals_strings[:-1])):
                v.append(eval(evals_strings[i]))
        cond = eval(evals_strings[-1])
    if exp_normalize and torch.all(cond != 0):
        cond = cond * cond_norm / cond.norm()
    # print(get_entropy(cond))
    return cond.to(device=default_device)

class advancedDynamicCFG:
    def __init__(self):
        self.last_cfg_ht_one = 8
        self.previous_cond_pred = None
        
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "model": ("MODEL",),

                                "automatic_cfg" : (["None", "soft", "hard", "hard_squared", "range"], {"default": "hard"},),

                                "skip_uncond" : ("BOOLEAN", {"default": True}),
                                "fake_uncond_start" : ("BOOLEAN", {"default": False}),
                                "uncond_sigma_start": ("FLOAT", {"default": 1000, "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                                "uncond_sigma_end":   ("FLOAT", {"default": 1, "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),

                                "lerp_uncond" : ("BOOLEAN", {"default": False}),
                                "lerp_uncond_strength":    ("FLOAT", {"default": 2, "min": 0.0, "max": 10.0, "step": 0.1, "round": 0.1}),
                                "lerp_uncond_sigma_start": ("FLOAT", {"default": 1000,  "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                                "lerp_uncond_sigma_end":   ("FLOAT", {"default": 1, "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),

                                "subtract_latent_mean" : ("BOOLEAN", {"default": False}),
                                "subtract_latent_mean_sigma_start": ("FLOAT", {"default": 1000,  "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                                "subtract_latent_mean_sigma_end":   ("FLOAT", {"default": 1, "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),

                                "latent_intensity_rescale"     : ("BOOLEAN", {"default": False}),
                                "latent_intensity_rescale_method" : (["soft","hard","range"], {"default": "hard"},),
                                "latent_intensity_rescale_cfg": ("FLOAT", {"default": 8,  "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.1}),
                                "latent_intensity_rescale_sigma_start": ("FLOAT", {"default": 1000,  "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                                "latent_intensity_rescale_sigma_end":   ("FLOAT", {"default": 3, "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),

                                "cond_exp": ("BOOLEAN", {"default": False}),
                                "cond_exp_normalize": ("BOOLEAN", {"default": False}),
                                "cond_exp_sigma_start": ("FLOAT", {"default": 1000,  "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                                "cond_exp_sigma_end":   ("FLOAT", {"default": 1,   "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                                "cond_exp_method": (["amplify", "root", "power", "erf", "erf_amplify", "exp_erf", "root_erf", "sine", "sine_exp", "sine_exp_diff", "sine_exp_diff_to_sine", "sine_root", "sine_root_diff", "sine_root_diff_to_sine", "theDaRkNeSs", "cosine", "sign", "zero", "previous_average", "eval",
                                                     "attention_modifiers_input_using_cond","attention_modifiers_input_using_uncond",
                                                     "subtract_attention_modifiers_input_using_cond","subtract_attention_modifiers_input_using_uncond"],),
                                "cond_exp_value": ("FLOAT", {"default": 2, "min": 0, "max": 100, "step": 0.1, "round": 0.01}),
                                
                                "uncond_exp": ("BOOLEAN", {"default": False}),
                                "uncond_exp_normalize": ("BOOLEAN", {"default": False}),
                                "uncond_exp_sigma_start": ("FLOAT", {"default": 1000,  "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                                "uncond_exp_sigma_end":   ("FLOAT", {"default": 1,   "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                                "uncond_exp_method": (["amplify", "root", "power", "erf", "erf_amplify", "exp_erf", "root_erf", "sine", "sine_exp", "sine_exp_diff", "sine_exp_diff_to_sine", "sine_root", "sine_root_diff", "sine_root_diff_to_sine", "theDaRkNeSs", "cosine", "sign", "zero", "previous_average", "eval",
                                                       "subtract_attention_modifiers_input_using_cond","subtract_attention_modifiers_input_using_uncond"],),
                                "uncond_exp_value": ("FLOAT", {"default": 2, "min": 0, "max": 100, "step": 0.1, "round": 0.01}),

                                "fake_uncond_exp": ("BOOLEAN", {"default": False}),
                                "fake_uncond_exp_normalize": ("BOOLEAN", {"default": False}),
                                "fake_uncond_exp_method" : (["cond_pred", "previous_average",
                                                             "amplify", "root", "power", "erf", "erf_amplify", "exp_erf", "root_erf", "sine", "sine_exp", "sine_exp_diff", "sine_exp_diff_to_sine", "sine_root", "sine_root_diff",
                                                             "sine_root_diff_to_sine", "theDaRkNeSs", "cosine", "sign", "zero", "eval",
                                                             "subtract_attention_modifiers_input_using_cond","subtract_attention_modifiers_input_using_uncond",
                                                             "attention_modifiers_input_using_cond","attention_modifiers_input_using_uncond"],),
                                "fake_uncond_exp_value": ("FLOAT", {"default": 2, "min": 0, "max": 1000, "step": 0.1, "round": 0.01}),
                                "fake_uncond_multiplier": ("INT", {"default": 1, "min": -1, "max": 1, "step": 1}),
                                "fake_uncond_sigma_start": ("FLOAT", {"default": 1000,  "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                                "fake_uncond_sigma_end": ("FLOAT", {"default": 1,  "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                                "auto_cfg_topk":    ("FLOAT", {"default": 0.25, "min": 0.0, "max": 0.5, "step": 0.05, "round": 0.01}),
                                "auto_cfg_ref":    ("FLOAT", {"default": 8, "min": 0.0, "max": 100, "step": 0.5, "round": 0.01}),
                                "attention_modifiers_global_enabled": ("BOOLEAN", {"default": False}),
                                "disable_cond": ("BOOLEAN", {"default": False}),
                                "disable_cond_sigma_start": ("FLOAT", {"default": 1000, "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                                "disable_cond_sigma_end":   ("FLOAT", {"default": 0, "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                                "save_as_preset": ("BOOLEAN", {"default": False}),
                                "preset_name": ("STRING", {"multiline": False}),
                              },
                              "optional":{
                                  "eval_string_cond": ("STRING", {"multiline": True}),
                                  "eval_string_uncond": ("STRING", {"multiline": True}),
                                  "eval_string_fake": ("STRING", {"multiline": True}),
                                  "args_filter": ("STRING", {"multiline": True, "forceInput": True}),
                                  "attention_modifiers_positive": ("ATTNMOD", {"forceInput": True}),
                                  "attention_modifiers_negative": ("ATTNMOD", {"forceInput": True}),
                                  "attention_modifiers_fake_negative": ("ATTNMOD", {"forceInput": True}),
                                  "attention_modifiers_global": ("ATTNMOD", {"forceInput": True}),
                              }
                              }
    RETURN_TYPES = ("MODEL","STRING",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Automatic_CFG"

    def patch(self, model, automatic_cfg = "None",
              skip_uncond = False, fake_uncond_start = False, uncond_sigma_start = 1000, uncond_sigma_end = 0,
              lerp_uncond = False, lerp_uncond_strength = 1, lerp_uncond_sigma_start = 1000, lerp_uncond_sigma_end = 1,
              subtract_latent_mean     = False,   subtract_latent_mean_sigma_start      = 1000, subtract_latent_mean_sigma_end     = 1,
              latent_intensity_rescale = False,   latent_intensity_rescale_sigma_start  = 1000, latent_intensity_rescale_sigma_end = 1,
              cond_exp = False, cond_exp_sigma_start  = 1000, cond_exp_sigma_end = 1000, cond_exp_method = "amplify", cond_exp_value = 2, cond_exp_normalize = False,
              uncond_exp = False, uncond_exp_sigma_start  = 1000, uncond_exp_sigma_end = 1000, uncond_exp_method = "amplify", uncond_exp_value = 2, uncond_exp_normalize = False,
              fake_uncond_exp = False, fake_uncond_exp_method = "amplify", fake_uncond_exp_value = 2, fake_uncond_exp_normalize = False, fake_uncond_multiplier = 1, fake_uncond_sigma_start = 1000, fake_uncond_sigma_end = 1,
              latent_intensity_rescale_cfg = 8, latent_intensity_rescale_method = "hard",
              ignore_pre_cfg_func = False, args_filter = "", auto_cfg_topk = 0.25, auto_cfg_ref = 8,
              eval_string_cond = "", eval_string_uncond = "", eval_string_fake = "",
              attention_modifiers_global_enabled = False,
              attention_modifiers_positive = [], attention_modifiers_negative = [], attention_modifiers_fake_negative = [], attention_modifiers_global = [],
              disable_cond=False, disable_cond_sigma_start=1000,disable_cond_sigma_end=1000, save_as_preset = False, preset_name = "", **kwargs
              ):
        
        # support_function()
        model_options_copy = deepcopy(model.model_options)
        monkey_patching_comfy_sampling_function()
        if args_filter != "":
            args_filter = args_filter.split(",")
        else:
            args_filter = [k for k, v in locals().items()]
        not_in_filter = ['self','model','args','args_filter','save_as_preset','preset_name','model_options_copy']
        if fake_uncond_exp_method != "eval":
            not_in_filter.append("eval_string")

        if save_as_preset and preset_name != "":
            preset_parameters = {key: value for key, value in locals().items() if key not in not_in_filter}
            with open(os.path.join(json_preset_path, preset_name+".json"), 'w', encoding='utf-8') as f:
                json.dump(preset_parameters, f)
            print(f"Preset saved with the name: {Fore.GREEN}{preset_name}{Fore.RESET}")
            print(f"{Fore.RED}Don't forget to turn the save toggle OFF to not overwrite!{Fore.RESET}")

        args_str = '\n'.join(f'{k}: {v}' for k, v in locals().items() if k not in not_in_filter and k in args_filter)

        sigmin, sigmax = get_sigmin_sigmax(model)

        lerp_start, lerp_end          = lerp_uncond_sigma_start, lerp_uncond_sigma_end
        subtract_start, subtract_end  = subtract_latent_mean_sigma_start, subtract_latent_mean_sigma_end
        rescale_start, rescale_end    = latent_intensity_rescale_sigma_start, latent_intensity_rescale_sigma_end
        print(f"Model maximum sigma: {sigmax} / Model minimum sigma: {sigmin}")
        m = model.clone()

        if skip_uncond or disable_cond:
            # set model_options sampler_pre_cfg_function
            m.model_options["sampler_pre_cfg_function"] = make_sampler_pre_cfg_function(uncond_sigma_end if skip_uncond else 0, uncond_sigma_start if skip_uncond else 100000,\
                                                                                        disable_cond_sigma_start if disable_cond else 100000, disable_cond_sigma_end if disable_cond else 100000)
            print(f"Sampling function patched. Uncond enabled from {round(uncond_sigma_start,2)} to {round(uncond_sigma_end,2)}")
        elif not ignore_pre_cfg_func:
            m.model_options.pop("sampler_pre_cfg_function", None)
            uncond_sigma_start, uncond_sigma_end = 1000000, 0

        top_k = auto_cfg_topk
        previous_cond_pred = None
        previous_sigma = None
        def automatic_cfg_function(args):
            nonlocal previous_sigma
            cond_scale = args["cond_scale"]
            input_x = args["input"]
            cond_pred = args["cond_denoised"]
            uncond_pred = args["uncond_denoised"]
            sigma = args["sigma"][0]
            model_options = args["model_options"]
            if self.previous_cond_pred is None:
                self.previous_cond_pred = cond_pred.clone().detach().to(device=cond_pred.device)
            if previous_sigma is None:
                previous_sigma = sigma.item()
            reference_cfg = auto_cfg_ref if auto_cfg_ref > 0 else cond_scale

            def fake_uncond_step():
                return fake_uncond_start and skip_uncond and (sigma > uncond_sigma_start or sigma < uncond_sigma_end) and sigma <= fake_uncond_sigma_start and sigma >= fake_uncond_sigma_end

            if fake_uncond_step():
                uncond_pred = cond_pred.clone().detach().to(device=cond_pred.device) * fake_uncond_multiplier

            if cond_exp and sigma <= cond_exp_sigma_start and sigma >= cond_exp_sigma_end:
                cond_pred = experimental_functions(cond_pred, cond_exp_method, cond_exp_value, cond_exp_normalize, self.previous_cond_pred, previous_sigma, sigma.item(), sigmax, attention_modifiers_positive, args, model_options_copy, eval_string_cond)
            if uncond_exp and sigma <= uncond_exp_sigma_start and sigma >= uncond_exp_sigma_end and not fake_uncond_step():
                uncond_pred = experimental_functions(uncond_pred, uncond_exp_method, uncond_exp_value, uncond_exp_normalize, self.previous_cond_pred, previous_sigma, sigma.item(), sigmax, attention_modifiers_negative, args, model_options_copy, eval_string_uncond)
            if fake_uncond_step() and fake_uncond_exp:
                uncond_pred = experimental_functions(uncond_pred, fake_uncond_exp_method, fake_uncond_exp_value, fake_uncond_exp_normalize, self.previous_cond_pred, previous_sigma, sigma.item(), sigmax, attention_modifiers_fake_negative, args, model_options_copy, eval_string_fake)
            self.previous_cond_pred = cond_pred.clone().detach().to(device=cond_pred.device)

            if sigma >= sigmax or cond_scale > 1:
                self.last_cfg_ht_one = cond_scale
            target_intensity = self.last_cfg_ht_one / 10

            if ((check_skip(sigma, uncond_sigma_start, uncond_sigma_end) and skip_uncond) and not fake_uncond_step()) or cond_scale == 1:
                return input_x - cond_pred

            if lerp_uncond and not check_skip(sigma, lerp_start, lerp_end) and lerp_uncond_strength != 1:
                uncond_pred_norm = uncond_pred.norm()
                uncond_pred = torch.lerp(cond_pred, uncond_pred, lerp_uncond_strength)
                uncond_pred = uncond_pred * uncond_pred_norm / uncond_pred.norm()
            cond   = input_x - cond_pred
            uncond = input_x - uncond_pred

            if automatic_cfg == "None":
                return uncond + cond_scale * (cond - uncond)

            denoised_tmp = input_x - (uncond + reference_cfg * (cond - uncond))

            for b in range(len(denoised_tmp)):
                denoised_ranges = get_denoised_ranges(denoised_tmp[b], automatic_cfg, top_k)
                for c in range(len(denoised_tmp[b])):
                    fixeds_scale = reference_cfg * target_intensity / denoised_ranges[c]
                    denoised_tmp[b][c] = uncond[b][c] + fixeds_scale * (cond[b][c] - uncond[b][c])

            return denoised_tmp

        def center_mean_latent_post_cfg(args):
            denoised = args["denoised"]
            sigma    = args["sigma"][0]
            if check_skip(sigma, subtract_start, subtract_end):
                return denoised
            denoised = center_latent_mean_values(denoised, False, 1)
            return denoised

        def rescale_post_cfg(args):
            denoised   = args["denoised"]
            sigma      = args["sigma"][0]

            if check_skip(sigma, rescale_start, rescale_end):
                return denoised
            target_intensity = latent_intensity_rescale_cfg / 10
            for b in range(len(denoised)):
                denoised_ranges = get_denoised_ranges(denoised[b], latent_intensity_rescale_method)
                for c in range(len(denoised[b])):
                    scale_correction = target_intensity / denoised_ranges[c]
                    denoised[b][c]   = denoised[b][c] * scale_correction
            return denoised
        
        tmp_model_options = deepcopy(m.model_options)
        if attention_modifiers_global_enabled:
            # print(f"{Fore.GREEN}Sigma timings are ignored for global modifiers.{Fore.RESET}")
            for atm in attention_modifiers_global:
                block_layers = {"input": atm['unet_block_id_input'], "middle": atm['unet_block_id_middle'], "output": atm['unet_block_id_output']}
                for unet_block in block_layers:
                    for unet_block_id in block_layers[unet_block].split(","):
                        if unet_block_id != "":
                            unet_block_id = int(unet_block_id)
                            tmp_model_options = set_model_options_patch_replace(tmp_model_options, attention_modifier(atm['self_attn_mod_eval']).modified_attention, atm['unet_attn'], unet_block, unet_block_id)
            m.model_options = tmp_model_options

        if not ignore_pre_cfg_func:
            m.set_model_sampler_cfg_function(automatic_cfg_function, disable_cfg1_optimization = False)
        if subtract_latent_mean:
            m.set_model_sampler_post_cfg_function(center_mean_latent_post_cfg)
        if latent_intensity_rescale:
            m.set_model_sampler_post_cfg_function(rescale_post_cfg)
        return (m, args_str, )

class attentionModifierParametersNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "sigma_start": ("FLOAT", {"default": 1000, "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                                "sigma_end":   ("FLOAT", {"default":  0, "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                                "self_attn_mod_eval":   ("STRING", {"multiline": True }, {"default": ""}),
                                "unet_block_id_input":  ("STRING", {"multiline": False}, {"default": ""}),
                                "unet_block_id_middle": ("STRING", {"multiline": False}, {"default": ""}),
                                "unet_block_id_output": ("STRING", {"multiline": False}, {"default": ""}),
                                "unet_attn": (["attn1","attn2","both"],),
                              },
                              "optional":{
                                  "join_parameters": ("ATTNMOD", {"forceInput": True}),
                              }}
    
    RETURN_TYPES = ("ATTNMOD","STRING",)
    RETURN_NAMES = ("Attention modifier", "Parameters as string")
    FUNCTION = "exec"
    CATEGORY = "model_patches/Automatic_CFG/experimental_attention_modifiers"
    def exec(self, join_parameters=None, **kwargs):
        info_string = "\n".join([f"{k}: {v}" for k,v in kwargs.items() if v != ""])
        if kwargs['unet_attn'] == "both":
            copy_kwargs = kwargs.copy()
            kwargs['unet_attn'] = "attn1"
            copy_kwargs['unet_attn'] = "attn2"
            out_modifiers = [kwargs, copy_kwargs]
        else:
            out_modifiers = [kwargs]
        return (out_modifiers if join_parameters is None else join_parameters + out_modifiers, info_string, )

class attentionModifierBruteforceParametersNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                                "sigma_start": ("FLOAT", {"default": 1000, "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                                "sigma_end":   ("FLOAT", {"default":  0, "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                                "self_attn_mod_eval":   ("STRING", {"multiline": True , "default": ""}),
                                "unet_block_id_input":  ("STRING", {"multiline": False, "default": "4,5,7,8"}),
                                "unet_block_id_middle": ("STRING", {"multiline": False, "default": "0"}),
                                "unet_block_id_output": ("STRING", {"multiline": False, "default": "0,1,2,3,4,5"}),
                                "unet_attn": (["attn1","attn2","both"],),
                              },
                              "optional":{
                                  "join_parameters": ("ATTNMOD", {"forceInput": True}),
                              }}
    
    RETURN_TYPES = ("ATTNMOD","STRING",)
    RETURN_NAMES = ("Attention modifier", "Parameters as string")
    FUNCTION = "exec"
    CATEGORY = "model_patches/Automatic_CFG/experimental_attention_modifiers"

    def create_sequence_parameters(self, input_str, middle_str, output_str):
        input_values  = input_str.split(",")  if input_str  else []
        middle_values = middle_str.split(",") if middle_str else []
        output_values = output_str.split(",") if output_str else []
        result = []
        result.extend([{"unet_block_id_input": val, "unet_block_id_middle": "", "unet_block_id_output": ""} for val in input_values])
        result.extend([{"unet_block_id_input": "", "unet_block_id_middle": val, "unet_block_id_output": ""} for val in middle_values])
        result.extend([{"unet_block_id_input": "", "unet_block_id_middle": "", "unet_block_id_output": val} for val in output_values])
        return result

    def exec(self, seed, join_parameters=None, **kwargs):
        sequence_parameters = self.create_sequence_parameters(kwargs['unet_block_id_input'],kwargs['unet_block_id_middle'],kwargs['unet_block_id_output'])
        lenseq = len(sequence_parameters)
        current_index = seed % lenseq
        current_sequence = sequence_parameters[current_index]
        kwargs["unet_block_id_input"]  = current_sequence["unet_block_id_input"]
        kwargs["unet_block_id_middle"] = current_sequence["unet_block_id_middle"]
        kwargs["unet_block_id_output"] = current_sequence["unet_block_id_output"]
        if current_sequence["unet_block_id_input"] != "":
            current_block_string = f"unet_block_id_input: {current_sequence['unet_block_id_input']}"
        elif current_sequence["unet_block_id_middle"] != "":
            current_block_string = f"unet_block_id_middle: {current_sequence['unet_block_id_middle']}"
        elif current_sequence["unet_block_id_output"] != "":
            current_block_string = f"unet_block_id_output: {current_sequence['unet_block_id_output']}"
        info_string = f"Progress: {current_index+1}/{lenseq}\n{kwargs['self_attn_mod_eval']}\n{kwargs['unet_attn']} {current_block_string}"
        if kwargs['unet_attn'] == "both":
            copy_kwargs = kwargs.copy()
            kwargs['unet_attn'] = "attn1"
            copy_kwargs['unet_attn'] = "attn2"
            out_modifiers = [kwargs, copy_kwargs]
        else:
            out_modifiers = [kwargs]
        return (out_modifiers if join_parameters is None else join_parameters + out_modifiers, info_string, )
    
class attentionModifierConcatNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "parameters_1": ("ATTNMOD", {"forceInput": True}),
                                "parameters_2": ("ATTNMOD", {"forceInput": True}),
                              }}
    
    RETURN_TYPES = ("ATTNMOD",)
    FUNCTION = "exec"
    CATEGORY = "model_patches/Automatic_CFG/experimental_attention_modifiers"
    def exec(self, parameters_1, parameters_2):
        output_parms = parameters_1 + parameters_2
        return (output_parms, )

class simpleDynamicCFG:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "model": ("MODEL",),
                                "hard_mode" : ("BOOLEAN", {"default": True}),
                                "boost" : ("BOOLEAN", {"default": True}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Automatic_CFG/presets"

    def patch(self, model, hard_mode, boost):
        advcfg = advancedDynamicCFG()
        m = advcfg.patch(model,
                         skip_uncond = boost,
                         uncond_sigma_start = 1000,  uncond_sigma_end = 1,
                         automatic_cfg = "hard" if hard_mode else "soft"
                         )[0]
        return (m, )

class presetLoader:
    @classmethod
    def INPUT_TYPES(s):
        presets_files = [pj.replace(".json","") for pj in os.listdir(json_preset_path) if ".json" in pj and pj not in ["Experimental_temperature.json","do_not_delete.json"]]
        presets_files = sorted(presets_files, key=str.lower)
        return {"required": {
                                "model": ("MODEL",),
                                "preset" : (presets_files, {"default": "Excellent_attention"}),
                                "uncond_sigma_end":   ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                                "use_uncond_sigma_end_from_preset" : ("BOOLEAN", {"default": True}),
                                "automatic_cfg" : (["From preset","None", "soft", "hard", "hard_squared", "range"],),
                              },
                              "optional":{
                                  "join_global_parameters": ("ATTNMOD", {"forceInput": True}),
                              }}
    RETURN_TYPES = ("MODEL", "STRING", "STRING",)
    RETURN_NAMES = ("Model", "Preset name", "Parameters as string",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Automatic_CFG"

    def patch(self, model, preset, uncond_sigma_end, use_uncond_sigma_end_from_preset, automatic_cfg, join_global_parameters=None):
        with open(os.path.join(json_preset_path, preset+".json"), 'r', encoding='utf-8') as f:
            preset_args = json.load(f)
        if not use_uncond_sigma_end_from_preset:
            preset_args["uncond_sigma_end"] = uncond_sigma_end
            preset_args["fake_uncond_sigma_end"] = uncond_sigma_end
            preset_args["fake_uncond_exp_sigma_end"] = uncond_sigma_end
            preset_args["uncond_exp_sigma_end"] = uncond_sigma_end            
        
        if join_global_parameters is not None:
            preset_args["attention_modifiers_global"] = preset_args["attention_modifiers_global"] + join_global_parameters
            preset_args["attention_modifiers_global_enabled"] = True

        if automatic_cfg != "From preset":
            preset_args["automatic_cfg"] = automatic_cfg
        
        advcfg = advancedDynamicCFG()
        m = advcfg.patch(model, **preset_args)[0]
        info_string = ",\n".join([f"\"{k}\": {v}" for k,v in preset_args.items() if v != ""])
        print(f"Preset {Fore.GREEN}{preset}{Fore.RESET} loaded successfully!")
        return (m, preset, info_string,)

class simpleDynamicCFGlerpUncond:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "model": ("MODEL",),
                                "boost" : ("BOOLEAN", {"default": True}),
                                "negative_strength": ("FLOAT", {"default": 1, "min": 0.0, "max": 5.0, "step": 0.1, "round": 0.1}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Automatic_CFG/presets"

    def patch(self, model, boost, negative_strength):
        advcfg = advancedDynamicCFG()
        m = advcfg.patch(model=model,
                         automatic_cfg="hard", skip_uncond=boost,
                         uncond_sigma_start = 15, uncond_sigma_end = 1,
                         lerp_uncond=negative_strength != 1, lerp_uncond_strength=negative_strength,
                         lerp_uncond_sigma_start = 15, lerp_uncond_sigma_end = 1
                         )[0]
        return (m, )

class postCFGrescaleOnly:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "model": ("MODEL",),
                                "subtract_latent_mean" : ("BOOLEAN", {"default": True}),
                                "subtract_latent_mean_sigma_start": ("FLOAT", {"default": 1000,  "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.1}),
                                "subtract_latent_mean_sigma_end":   ("FLOAT", {"default": 7.5, "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.1}),
                                "latent_intensity_rescale"     : ("BOOLEAN", {"default": True}),
                                "latent_intensity_rescale_method" : (["soft","hard","range"], {"default": "hard"},),
                                "latent_intensity_rescale_cfg" : ("FLOAT", {"default": 8,  "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.1}),
                                "latent_intensity_rescale_sigma_start": ("FLOAT", {"default": 1000,  "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.1}),
                                "latent_intensity_rescale_sigma_end":   ("FLOAT", {"default": 5, "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.1}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Automatic_CFG/utils"

    def patch(self, model,
              subtract_latent_mean, subtract_latent_mean_sigma_start, subtract_latent_mean_sigma_end,
              latent_intensity_rescale, latent_intensity_rescale_method, latent_intensity_rescale_cfg, latent_intensity_rescale_sigma_start, latent_intensity_rescale_sigma_end
              ):
        advcfg = advancedDynamicCFG()
        m = advcfg.patch(model=model, 
                         subtract_latent_mean = subtract_latent_mean,
                         subtract_latent_mean_sigma_start = subtract_latent_mean_sigma_start, subtract_latent_mean_sigma_end = subtract_latent_mean_sigma_end,
                         latent_intensity_rescale = latent_intensity_rescale, latent_intensity_rescale_cfg = latent_intensity_rescale_cfg, latent_intensity_rescale_method = latent_intensity_rescale_method,
                         latent_intensity_rescale_sigma_start = latent_intensity_rescale_sigma_start, latent_intensity_rescale_sigma_end = latent_intensity_rescale_sigma_end,
                         ignore_pre_cfg_func = True
                         )[0]
        return (m, )

class simpleDynamicCFGHighSpeed:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "model": ("MODEL",),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Automatic_CFG/presets"

    def patch(self, model):
        advcfg = advancedDynamicCFG()
        m = advcfg.patch(model=model, automatic_cfg = "hard",
                         skip_uncond = True, uncond_sigma_start = 7.5, uncond_sigma_end = 1)[0]
        return (m, )

class simpleDynamicCFGwarpDrive:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "model": ("MODEL",),
                                "uncond_sigma_start": ("FLOAT", {"default": 5.5, "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                                "uncond_sigma_end":   ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                                "fake_uncond_sigma_end": ("FLOAT", {"default": 1.0,  "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Automatic_CFG/presets"

    def patch(self, model, uncond_sigma_start, uncond_sigma_end, fake_uncond_sigma_end):
        advcfg = advancedDynamicCFG()
        print(f"                                                {Fore.CYAN}WARP DRIVE MODE ENGAGED!{Style.RESET_ALL}\n    Settings suggestions:\n"
            f"              {Fore.GREEN}1/1/1:   {Fore.YELLOW}Maaaxxxiiimum speeeeeed.{Style.RESET_ALL} {Fore.RED}Uncond disabled.{Style.RESET_ALL} {Fore.MAGENTA}Fasten your seatbelt!{Style.RESET_ALL}\n"
            f"              {Fore.GREEN}3/1/1:   {Fore.YELLOW}Risky space-time continuum distortion.{Style.RESET_ALL} {Fore.MAGENTA}Awesome for prompts with a clear subject!{Style.RESET_ALL}\n"
            f"              {Fore.GREEN}5.5/1/1: {Fore.YELLOW}Frameshift Drive Autopilot: {Fore.GREEN}Engaged.{Style.RESET_ALL} {Fore.MAGENTA}Should work with anything but do it better and faster!{Style.RESET_ALL}")

        m = advcfg.patch(model=model, automatic_cfg = "hard",
                         skip_uncond = True, uncond_sigma_start = uncond_sigma_start, uncond_sigma_end = uncond_sigma_end,
                         fake_uncond_sigma_end = fake_uncond_sigma_end, fake_uncond_sigma_start = 1000, fake_uncond_start=True,
                         fake_uncond_exp=True,fake_uncond_exp_normalize=True,fake_uncond_exp_method="previous_average",
                         cond_exp = False, cond_exp_sigma_start  = 9, cond_exp_sigma_end = uncond_sigma_start, cond_exp_method = "erf", cond_exp_normalize = True,
                         )[0]
        return (m, )

class simpleDynamicCFGunpatch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "model": ("MODEL",),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "unpatch"

    CATEGORY = "model_patches/Automatic_CFG/utils"

    def unpatch(self, model):
        m = model.clone()
        m.model_options.pop("sampler_pre_cfg_function", None)
        return (m, )

class simpleDynamicCFGExcellentattentionPatch:
    @classmethod
    def INPUT_TYPES(s):
        inputs = {"required": {
                        "model": ("MODEL",),
                        "Auto_CFG": ("BOOLEAN", {"default": True}),
                        "patch_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 1.0, "round": 0.01}),
                        "patch_cond":   ("BOOLEAN", {"default": True}),
                        "patch_uncond": ("BOOLEAN", {"default": True}),
                        "light_patch":  ("BOOLEAN", {"default": False}),
                        "mute_self_input_layer_8_cond":    ("BOOLEAN", {"default": False}),
                        "mute_cross_input_layer_8_cond":   ("BOOLEAN", {"default": False}),
                        "mute_self_input_layer_8_uncond":  ("BOOLEAN", {"default": True}),
                        "mute_cross_input_layer_8_uncond": ("BOOLEAN", {"default": False}),
                        "uncond_sigma_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                        "bypass_layer_8_instead_of_mute": ("BOOLEAN", {"default": False}),
                        "save_as_preset": ("BOOLEAN", {"default": False}),
                        "preset_name":    ("STRING",  {"multiline": False}),
                        },
                        "optional":{
                                "attn_mod_for_positive_operation": ("ATTNMOD", {"forceInput": True}),
                                "attn_mod_for_negative_operation": ("ATTNMOD", {"forceInput": True}),
                        },
                    }
        if "dev_env.txt" in os.listdir(current_dir):
            inputs['optional'].update({"attn_mod_for_global_operation": ("ATTNMOD", {"forceInput": True})})
        return inputs
    
    RETURN_TYPES = ("MODEL","STRING",)
    RETURN_NAMES = ("Model", "Parameters as string",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Automatic_CFG"

    def patch(self, model, Auto_CFG, patch_multiplier, patch_cond, patch_uncond, light_patch,
              mute_self_input_layer_8_cond, mute_cross_input_layer_8_cond,
              mute_self_input_layer_8_uncond, mute_cross_input_layer_8_uncond,
              uncond_sigma_end,bypass_layer_8_instead_of_mute, save_as_preset, preset_name,
              attn_mod_for_positive_operation = None, attn_mod_for_negative_operation = None, attn_mod_for_global_operation = None):
        
        parameters_as_string = "Excellent attention:\n" + "\n".join([f"{k}: {v}" for k, v in locals().items() if k not in ["self", "model"]])
        
        with open(os.path.join(json_preset_path, "Excellent_attention.json"), 'r', encoding='utf-8') as f:
            patch_parameters = json.load(f)

        attn_patch = {"sigma_start": 1000, "sigma_end": 0,
                       "self_attn_mod_eval": f"normalize_tensor(q+(q-attention_basic(attnbc, k, v, extra_options['n_heads'])))*attnbc.norm()*{patch_multiplier}",
                       "unet_block_id_input": "", "unet_block_id_middle": "0", "unet_block_id_output": "", "unet_attn": "attn2"}
        attn_patch_light = {"sigma_start": 1000, "sigma_end": 0,
                       "self_attn_mod_eval": f"q*{patch_multiplier}",
                       "unet_block_id_input": "", "unet_block_id_middle": "0", "unet_block_id_output": "", "unet_attn": "attn2"}

        kill_self_input_8 = {
            "sigma_start": 1000,
            "sigma_end": 0,
            "self_attn_mod_eval": "q" if bypass_layer_8_instead_of_mute else "torch.zeros_like(q)",
            "unet_block_id_input": "8",
            "unet_block_id_middle": "",
            "unet_block_id_output": "",
            "unet_attn": "attn1"}
        
        kill_cross_input_8 = kill_self_input_8.copy()
        kill_cross_input_8['unet_attn'] = "attn2"
        
        attention_modifiers_positive = []
        attention_modifiers_fake_negative = []

        if patch_cond: attention_modifiers_positive.append(attn_patch) if not light_patch else attention_modifiers_positive.append(attn_patch_light)
        if mute_self_input_layer_8_cond:  attention_modifiers_positive.append(kill_self_input_8)
        if mute_cross_input_layer_8_cond: attention_modifiers_positive.append(kill_cross_input_8)

        if patch_uncond: attention_modifiers_fake_negative.append(attn_patch) if not light_patch else attention_modifiers_fake_negative.append(attn_patch_light)
        if mute_self_input_layer_8_uncond:  attention_modifiers_fake_negative.append(kill_self_input_8)
        if mute_cross_input_layer_8_uncond: attention_modifiers_fake_negative.append(kill_cross_input_8)

        patch_parameters['attention_modifiers_positive']      = attention_modifiers_positive
        patch_parameters['attention_modifiers_fake_negative'] = attention_modifiers_fake_negative

        if attn_mod_for_positive_operation is not None:
            patch_parameters['attention_modifiers_positive'] = patch_parameters['attention_modifiers_positive'] + attn_mod_for_positive_operation
        if attn_mod_for_negative_operation is not None:
            patch_parameters['attention_modifiers_fake_negative'] = patch_parameters['attention_modifiers_fake_negative'] + attn_mod_for_negative_operation
        if attn_mod_for_global_operation is not None:
            patch_parameters["attention_modifiers_global_enabled"] = True
            patch_parameters['attention_modifiers_global'] = attn_mod_for_global_operation

        patch_parameters["uncond_sigma_end"]      = uncond_sigma_end
        patch_parameters["fake_uncond_sigma_end"] = uncond_sigma_end
        patch_parameters["automatic_cfg"] = "hard" if Auto_CFG else "None"

        if save_as_preset:
            patch_parameters["save_as_preset"] = save_as_preset
            patch_parameters["preset_name"] = preset_name

        advcfg = advancedDynamicCFG()
        m = advcfg.patch(model, **patch_parameters)[0]
        
        return (m, parameters_as_string, )
    
class simpleDynamicCFGCustomAttentionPatch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "model": ("MODEL",),
                                "Auto_CFG": ("BOOLEAN", {"default": True}),
                                "cond_mode" :   (["replace_by_custom","normal+(normal-custom_cond)*multiplier","normal+(normal-custom_uncond)*multiplier"],),
                                "uncond_mode" : (["replace_by_custom","normal+(normal-custom_cond)*multiplier","normal+(normal-custom_uncond)*multiplier"],),
                                "cond_diff_multiplier":   ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                                "uncond_diff_multiplier": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                                "uncond_sigma_end":       ("FLOAT", {"default": 1.0, "min":    0.0, "max": 10000, "step": 0.1, "round": 0.01}),
                                "save_as_preset": ("BOOLEAN", {"default": False}),
                                "preset_name":    ("STRING",  {"multiline": False}),
                              },
                              "optional":{
                                  "attn_mod_for_positive_operation": ("ATTNMOD", {"forceInput": True}),
                                  "attn_mod_for_negative_operation": ("ATTNMOD", {"forceInput": True}),
                              }}
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("Model",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Automatic_CFG/experimental_attention_modifiers"

    def patch(self, model, Auto_CFG, cond_mode, uncond_mode, cond_diff_multiplier, uncond_diff_multiplier, uncond_sigma_end, save_as_preset, preset_name,
              attn_mod_for_positive_operation = [], attn_mod_for_negative_operation = []):
                
        with open(os.path.join(json_preset_path, "do_not_delete.json"), 'r', encoding='utf-8') as f:
            patch_parameters = json.load(f)
        
        patch_parameters["cond_exp_value"]   = cond_diff_multiplier
        patch_parameters["uncond_exp_value"] = uncond_diff_multiplier

        if cond_mode != "replace_by_custom":
            patch_parameters["disable_cond"] = False
        if cond_mode == "normal+(normal-custom_cond)*multiplier":
            patch_parameters["cond_exp_method"] = "subtract_attention_modifiers_input_using_cond"
        elif cond_mode == "normal+(normal-custom_uncond)*multiplier":
            patch_parameters["cond_exp_method"] = "subtract_attention_modifiers_input_using_uncond"

        if uncond_mode != "replace_by_custom":
            patch_parameters["uncond_sigma_start"] = 1000.0
            patch_parameters["fake_uncond_exp"]    = False
            patch_parameters["uncond_exp"]         = True
        
        if uncond_mode == "normal+(normal-custom_cond)*multiplier":
            patch_parameters["uncond_exp_method"] = "subtract_attention_modifiers_input_using_cond"
        elif uncond_mode == "normal+(normal-custom_uncond)*multiplier":
            patch_parameters["uncond_exp_method"] = "subtract_attention_modifiers_input_using_uncond"

        if cond_mode != "replace_by_custom" and attn_mod_for_positive_operation != []:
            smallest_sigma = min([float(x['sigma_end']) for x in attn_mod_for_positive_operation])
            patch_parameters["disable_cond_sigma_end"] = smallest_sigma
            patch_parameters["cond_exp_sigma_end"]     = smallest_sigma

        if uncond_mode != "replace_by_custom" and attn_mod_for_negative_operation != []:
            smallest_sigma = min([float(x['sigma_end'])   for x in attn_mod_for_negative_operation])
            patch_parameters["uncond_exp_sigma_end"] = smallest_sigma
            patch_parameters["fake_uncond_start"]    = False
        # else:
        #     biggest_sigma = max([float(x['sigma_start']) for x in attn_mod_for_negative_operation])
        #     patch_parameters["fake_uncond_sigma_start"] = biggest_sigma

        patch_parameters["automatic_cfg"] = "hard" if Auto_CFG else "None"
        patch_parameters['attention_modifiers_positive']      = attn_mod_for_positive_operation
        patch_parameters['attention_modifiers_negative']      = attn_mod_for_negative_operation
        patch_parameters['attention_modifiers_fake_negative'] = attn_mod_for_negative_operation
        patch_parameters["uncond_sigma_end"]      = uncond_sigma_end
        patch_parameters["fake_uncond_sigma_end"] = uncond_sigma_end
        patch_parameters["save_as_preset"] = save_as_preset
        patch_parameters["preset_name"]    = preset_name
        
        advcfg = advancedDynamicCFG()
        m = advcfg.patch(model, **patch_parameters)[0]
        
        return (m, )




class attentionModifierSingleLayerBypassNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "sigma_start": ("FLOAT", {"default": 1000, "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                                "sigma_end":   ("FLOAT", {"default":  0, "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                                "block_name":  (["input","middle","output"],),
                                "block_number": ("INT", {"default": 0, "min": 0, "max": 12, "step": 1}),
                                "unet_attn": (["attn1","attn2","both"],),
                              },
                              "optional":{
                                  "join_parameters": ("ATTNMOD", {"forceInput": True}),
                              }}
    
    RETURN_TYPES = ("ATTNMOD","STRING",)
    RETURN_NAMES = ("Attention modifier", "Parameters as string")
    FUNCTION = "exec"
    CATEGORY = "model_patches/Automatic_CFG/experimental_attention_modifiers"

    def exec(self, sigma_start, sigma_end, block_name, block_number, unet_attn, join_parameters=None):
        attn_modifier_dict = {
        "sigma_start": sigma_start, "sigma_end": sigma_end,
         "self_attn_mod_eval": "q",
         "unet_block_id_input":  str(block_number) if block_name == "input" else "",
         "unet_block_id_middle": str(block_number) if block_name == "middle" else "",
         "unet_block_id_output": str(block_number) if block_name == "output" else "",
         "unet_attn": f"{unet_attn}"
         }

        info_string = "\n".join([f"{k}: {v}" for k,v in attn_modifier_dict.items() if v != ""])

        if unet_attn == "both":
            attn_modifier_dict['unet_attn'] = "attn1"
            copy_attn_modifier_dict = attn_modifier_dict.copy()
            copy_attn_modifier_dict['unet_attn'] = "attn2"
            out_modifiers = [attn_modifier_dict, copy_attn_modifier_dict]
        else:
            out_modifiers = [attn_modifier_dict]

        return (out_modifiers if join_parameters is None else join_parameters + out_modifiers, info_string, )

class attentionModifierSingleLayerTemperatureNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "sigma_start": ("FLOAT", {"default": 1000, "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                                "sigma_end":   ("FLOAT", {"default":  0, "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                                "block_name":  (["input","middle","output"],),
                                "block_number": ("INT", {"default": 0, "min": 0, "max": 12, "step": 1}),
                                "unet_attn": (["attn1","attn2","both"],),
                                "temperature": ("FLOAT", {"default":  1, "min": 0.0, "max": 10000.0, "step": 0.01, "round": 0.01}),
                              },
                              "optional":{
                                  "join_parameters": ("ATTNMOD", {"forceInput": True}),
                              }}
    
    RETURN_TYPES = ("ATTNMOD","STRING",)
    RETURN_NAMES = ("Attention modifier", "Parameters as string")
    FUNCTION = "exec"
    CATEGORY = "model_patches/Automatic_CFG/experimental_attention_modifiers"

    def exec(self, sigma_start, sigma_end, block_name, block_number, unet_attn, temperature, join_parameters=None):
        attn_modifier_dict = {
        "sigma_start": sigma_start, "sigma_end": sigma_end,
         "self_attn_mod_eval": f"temperature_patcher({temperature}).attention_basic_with_temperature(q, k, v, extra_options)",
         "unet_block_id_input":  str(block_number) if block_name == "input" else "",
         "unet_block_id_middle": str(block_number) if block_name == "middle" else "",
         "unet_block_id_output": str(block_number) if block_name == "output" else "",
         "unet_attn": f"{unet_attn}"
         }

        info_string = "\n".join([f"{k}: {v}" for k,v in attn_modifier_dict.items() if v != ""])

        if unet_attn == "both":
            attn_modifier_dict['unet_attn'] = "attn1"
            copy_attn_modifier_dict = attn_modifier_dict.copy()
            copy_attn_modifier_dict['unet_attn'] = "attn2"
            out_modifiers = [attn_modifier_dict, copy_attn_modifier_dict]
        else:
            out_modifiers = [attn_modifier_dict]

        return (out_modifiers if join_parameters is None else join_parameters + out_modifiers, info_string, )

class uncondZeroNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "model": ("MODEL",),
                                "scale": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 10.0, "step": 0.01, "round": 0.01}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/Automatic_CFG"

    def patch(self, model, scale):
        def custom_patch(args):
            cond_pred = args["cond_denoised"]
            input_x = args["input"]
            if args["sigma"][0] <= 1:
                return input_x - cond_pred
            cond   = input_x - cond_pred
            uncond = input_x - torch.zeros_like(cond)
            return uncond + scale * (cond - uncond)

        m = model.clone()
        m.set_model_sampler_cfg_function(custom_patch)
        return (m, )
