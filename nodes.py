from copy import deepcopy
import comfy.samplers
import numpy as np
import torch
import math
import torch.nn.functional as F

original_sampling_function = deepcopy(comfy.samplers.sampling_function)
minimum_sigma_to_disable_uncond = 0
maximum_sigma_to_enable_uncond  = 1000000
global_skip_uncond = False

def sampling_function_patched(model, x, timestep, uncond, cond, cond_scale, model_options={}, seed=None):
        if math.isclose(cond_scale, 1.0) and model_options.get("disable_cfg1_optimization", False) == False or ((timestep[0] < minimum_sigma_to_disable_uncond or timestep[0] > maximum_sigma_to_enable_uncond) and global_skip_uncond):
            uncond_ = None
        else:
            uncond_ = uncond

        conds = [cond, uncond_]

        out = comfy.samplers.calc_cond_batch(model, conds, x, timestep, model_options)
        cond_pred = out[0]
        uncond_pred = out[1]

        if "sampler_cfg_function" in model_options:
            args = {"cond": x - cond_pred, "uncond": x - uncond_pred, "cond_scale": cond_scale, "timestep": timestep, "input": x, "sigma": timestep,
                    "cond_denoised": cond_pred, "uncond_denoised": uncond_pred, "model": model, "model_options": model_options, "cond_pos": cond, "cond_neg": uncond}
            cfg_result = x - model_options["sampler_cfg_function"](args)
        else:
            cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale

        for fn in model_options.get("sampler_post_cfg_function", []):
            args = {"denoised": cfg_result, "cond": cond, "uncond": uncond, "model": model, "uncond_denoised": uncond_pred, "cond_denoised": cond_pred,
                    "sigma": timestep, "model_options": model_options, "input": x}
            cfg_result = fn(args)

        return cfg_result

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

def get_sigmas_start_end(sigmin, sigmax, start_percentage, end_percentage):
    high_sigma_threshold = (sigmax - sigmin) / 100 * start_percentage
    low_sigma_threshold  = (sigmax - sigmin) / 100 * end_percentage
    return high_sigma_threshold, low_sigma_threshold

def gaussian_similarity(x, y, sigma=1.0):
    diff = (x - y) ** 2
    return torch.exp(-diff / (2 * sigma ** 2))
    
def check_skip(sigma, high_sigma_threshold, low_sigma_threshold):
    return sigma > high_sigma_threshold or sigma < low_sigma_threshold

def gaussian_kernel(size, sigma):
    ax = torch.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / kernel.sum()

def blur_tensor(input_tensor, sigma=2, kernel_size=7):
    device = input_tensor.device
    kernel = gaussian_kernel(kernel_size, sigma).unsqueeze(0).unsqueeze(0).to(device).to(input_tensor[0][0].dtype)
    padding = kernel_size // 2
    blurred_batch = []
    for batch in input_tensor:  # Iterate over each batch
        blurred_channels = []
        for channel in batch:  # Iterate over each channel
            blurred_channel = F.conv2d(channel.unsqueeze(0).unsqueeze(0), kernel, padding=padding)
            blurred_channels.append(blurred_channel.squeeze(0).squeeze(0))  # Corrected squeezing step
        blurred_batch.append(torch.stack(blurred_channels))
    return torch.stack(blurred_batch).to(device)

def square_and_norm(cond_input, method, exp_value, exp_normalize, pcp, psi, sigma, args, eval_string = ""):
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

    Using comfy.samplers.calc_cond_batch(args["model"], [args["cond_pos"], None], args["input"]-cond, args["timestep"], args["model_options"])[0]
    can work too.

    This whole mess has for initial goal to attempt to find the best way (or have some bruteforcing fun) to replace the uncond pred for as much as possible.
    """
    if method == "normal":
        return cond_input
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
    elif method == "previous_average":
        if sigma > 14:
            cond = torch.zeros_like(cond)
        else:
            cond = (pcp / psi * sigma + cond) / 2
    elif method == "eval":
        v = []
        evals_strings = eval_string.split(";")
        if len(evals_strings) > 1:
            for i in range(len(evals_strings[:-1])):
                v.append(eval(evals_strings[i]))
        cond = eval(evals_strings[-1])
    if exp_normalize and torch.all(cond != 0):
        cond = cond * cond_norm / cond.norm()
    # print(get_entropy(cond))
    return cond

class advancedDynamicCFG:
    def __init__(self):
        self.last_cfg_ht_one = 8

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "model": ("MODEL",),

                                "automatic_cfg" : (["None", "soft", "hard", "hard_squared", "range"], {"default": "hard"},),

                                "skip_uncond" : ("BOOLEAN", {"default": True}),
                                "fake_uncond_start" : ("BOOLEAN", {"default": False}),
                                "uncond_sigma_start": ("FLOAT", {"default": 5, "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                                "uncond_sigma_end":   ("FLOAT", {"default": 1, "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),

                                "lerp_uncond" : ("BOOLEAN", {"default": False}),
                                "lerp_uncond_strength":    ("FLOAT", {"default": 2, "min": 0.0, "max": 10.0, "step": 0.1, "round": 0.1}),
                                "lerp_uncond_sigma_start": ("FLOAT", {"default": 15,  "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                                "lerp_uncond_sigma_end":   ("FLOAT", {"default": 1, "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),

                                "subtract_latent_mean" : ("BOOLEAN", {"default": False}),
                                "subtract_latent_mean_sigma_start": ("FLOAT", {"default": 15,  "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                                "subtract_latent_mean_sigma_end":   ("FLOAT", {"default": 1, "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),

                                "latent_intensity_rescale"     : ("BOOLEAN", {"default": False}),
                                "latent_intensity_rescale_method" : (["soft","hard","range"], {"default": "hard"},),
                                "latent_intensity_rescale_cfg": ("FLOAT", {"default": 8,  "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.1}),
                                "latent_intensity_rescale_sigma_start": ("FLOAT", {"default": 15,  "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                                "latent_intensity_rescale_sigma_end":   ("FLOAT", {"default": 3, "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),

                                "cond_exp": ("BOOLEAN", {"default": False}),
                                "cond_exp_normalize": ("BOOLEAN", {"default": False}),
                                "cond_exp_sigma_start": ("FLOAT", {"default": 15,  "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                                "cond_exp_sigma_end":   ("FLOAT", {"default": 1,   "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                                "cond_exp_method": (["amplify", "root", "power", "erf", "erf_amplify", "exp_erf", "root_erf", "sine", "sine_exp", "sine_exp_diff", "sine_exp_diff_to_sine", "sine_root", "sine_root_diff", "sine_root_diff_to_sine", "theDaRkNeSs", "cosine", "sign", "zero"],),
                                "cond_exp_value": ("FLOAT", {"default": 2, "min": 1, "max": 100, "step": 0.1, "round": 0.01}),
                                
                                "uncond_exp": ("BOOLEAN", {"default": False}),
                                "uncond_exp_normalize": ("BOOLEAN", {"default": False}),
                                "uncond_exp_sigma_start": ("FLOAT", {"default": 15,  "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                                "uncond_exp_sigma_end":   ("FLOAT", {"default": 1,   "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                                "uncond_exp_method": (["normal", "amplify", "root", "power", "erf", "erf_amplify", "exp_erf", "root_erf", "sine", "sine_exp", "sine_exp_diff", "sine_exp_diff_to_sine", "sine_root", "sine_root_diff", "sine_root_diff_to_sine", "theDaRkNeSs", "cosine", "sign", "zero"],),
                                "uncond_exp_value": ("FLOAT", {"default": 2, "min": 1, "max": 100, "step": 0.1, "round": 0.01}),

                                "fake_uncond_exp": ("BOOLEAN", {"default": False}),
                                "fake_uncond_exp_normalize": ("BOOLEAN", {"default": False}),
                                "fake_uncond_exp_method" : (["normal", "previous_average", "amplify", "root", "power", "erf", "erf_amplify", "exp_erf", "root_erf", "sine", "sine_exp", "sine_exp_diff", "sine_exp_diff_to_sine", "sine_root", "sine_root_diff", "sine_root_diff_to_sine", "theDaRkNeSs", "cosine", "sign", "zero", "eval"],),
                                "fake_uncond_exp_value": ("FLOAT", {"default": 2, "min": 1, "max": 1000, "step": 0.1, "round": 0.01}),
                                "fake_uncond_multiplier": ("INT", {"default": 1, "min": -1, "max": 1, "step": 1}),
                                "fake_uncond_sigma_start": ("FLOAT", {"default": 15,  "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                                "fake_uncond_sigma_end": ("FLOAT", {"default": 5.5,  "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                              },
                              "optional":{
                                  "eval_string": ("STRING", {"multiline": True}),
                                  "args_filter": ("STRING", {"multiline": True, "forceInput": True})
                              }
                              }
    RETURN_TYPES = ("MODEL","STRING",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/automatic_cfg"

    def patch(self, model, automatic_cfg = "None",
              skip_uncond = False, fake_uncond_start = False, uncond_sigma_start = 15, uncond_sigma_end = 0,
              lerp_uncond = False, lerp_uncond_strength = 1, lerp_uncond_sigma_start = 15, lerp_uncond_sigma_end = 1,
              subtract_latent_mean     = False,   subtract_latent_mean_sigma_start      = 15, subtract_latent_mean_sigma_end     = 1,
              latent_intensity_rescale = False,   latent_intensity_rescale_sigma_start  = 15, latent_intensity_rescale_sigma_end = 1,
              cond_exp = False, cond_exp_sigma_start  = 15, cond_exp_sigma_end = 14, cond_exp_method = "amplify", cond_exp_value = 2, cond_exp_normalize = False,
              uncond_exp = False, uncond_exp_sigma_start  = 15, uncond_exp_sigma_end = 14, uncond_exp_method = "amplify", uncond_exp_value = 2, uncond_exp_normalize = False,
              fake_uncond_exp = False, fake_uncond_exp_method = "amplify", fake_uncond_exp_value = 2, fake_uncond_exp_normalize = False, fake_uncond_multiplier = 1, fake_uncond_sigma_start = 15, fake_uncond_sigma_end = 5.5,
              latent_intensity_rescale_cfg = 8, latent_intensity_rescale_method = "hard",
              ignore_pre_cfg_func = False, eval_string = "", args_filter = ""):

        args = locals()
        if args_filter != "":
            args_filter = args_filter.split(",")
        else:
            args_filter = [k for k, v in locals().items()]
        not_in_filter = ['self','model','args','args_filter']
        if fake_uncond_exp_method != "eval":
            not_in_filter.append("eval_string")
        args_str = '\n'.join(f'{k}: {v}' for k, v in locals().items() if k not in not_in_filter and k in args_filter)

        global minimum_sigma_to_disable_uncond, maximum_sigma_to_enable_uncond, global_skip_uncond
        sigmin, sigmax = get_sigmin_sigmax(model)
        
        lerp_start, lerp_end          = lerp_uncond_sigma_start, lerp_uncond_sigma_end
        subtract_start, subtract_end  = subtract_latent_mean_sigma_start, subtract_latent_mean_sigma_end
        rescale_start, rescale_end    = latent_intensity_rescale_sigma_start, latent_intensity_rescale_sigma_end
        print(f"Model maximum sigma: {sigmax} / Model minimum sigma: {sigmin}")
        if skip_uncond:
            global_skip_uncond = skip_uncond
            comfy.samplers.sampling_function = sampling_function_patched
            maximum_sigma_to_enable_uncond, minimum_sigma_to_disable_uncond = uncond_sigma_start, uncond_sigma_end
            print(f"Sampling function patched. Uncond enabled from {round(maximum_sigma_to_enable_uncond,2)} to {round(minimum_sigma_to_disable_uncond,2)}")
        elif not ignore_pre_cfg_func:
            global_skip_uncond = skip_uncond # just in case of mixup with another node
            comfy.samplers.sampling_function = original_sampling_function
            maximum_sigma_to_enable_uncond, minimum_sigma_to_disable_uncond = 1000000, 0
            print(f"Sampling function unpatched.")
        
        top_k = 0.25
        reference_cfg = 8
        previous_cond_pred = None
        previous_sigma = None
        def automatic_cfg_function(args):
            nonlocal previous_cond_pred, previous_sigma
            cond_scale = args["cond_scale"]
            input_x = args["input"]
            cond_pred = args["cond_denoised"]
            uncond_pred = args["uncond_denoised"]
            sigma = args["sigma"][0]
            model_options = args["model_options"]
            if previous_cond_pred is None:
                previous_cond_pred = deepcopy(cond_pred)
            if previous_sigma is None:
                previous_sigma = sigma.item()
            def fake_uncond_step():
                return fake_uncond_start and skip_uncond and (sigma > uncond_sigma_start or sigma < uncond_sigma_end) and sigma <= fake_uncond_sigma_start and sigma >= fake_uncond_sigma_end

            if fake_uncond_step():
                uncond_pred = cond_pred.clone() * fake_uncond_multiplier

            if cond_exp and sigma <= cond_exp_sigma_start and sigma >= cond_exp_sigma_end:
                cond_pred = square_and_norm(cond_pred, cond_exp_method, cond_exp_value, cond_exp_normalize, previous_cond_pred, previous_sigma, sigma.item(), args)
            if uncond_exp and sigma <= uncond_exp_sigma_start and sigma >= uncond_exp_sigma_end and not fake_uncond_step():
                uncond_pred = square_and_norm(uncond_pred, uncond_exp_method, uncond_exp_value, uncond_exp_normalize, previous_cond_pred, previous_sigma, sigma.item(), args)
            if fake_uncond_step() and fake_uncond_exp:
                uncond_pred = square_and_norm(uncond_pred, fake_uncond_exp_method, fake_uncond_exp_value, fake_uncond_exp_normalize, previous_cond_pred, previous_sigma, sigma.item(), args, eval_string)
            previous_cond_pred = deepcopy(cond_pred)

            if sigma >= sigmax or cond_scale > 1:
                self.last_cfg_ht_one = cond_scale
            target_intensity = self.last_cfg_ht_one / 10

            if ((check_skip(sigma, maximum_sigma_to_enable_uncond, minimum_sigma_to_disable_uncond) and skip_uncond) and not fake_uncond_step()) or cond_scale == 1:
                return input_x - cond_pred
            
            if lerp_uncond and not check_skip(sigma, lerp_start, lerp_end) and lerp_uncond_strength != 1:
                uncond_pred = torch.lerp(cond_pred, uncond_pred, lerp_uncond_strength)
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
        
        m = model.clone()
        if not ignore_pre_cfg_func:
            m.set_model_sampler_cfg_function(automatic_cfg_function, disable_cfg1_optimization = False)
        if subtract_latent_mean:
            m.set_model_sampler_post_cfg_function(center_mean_latent_post_cfg)
        if latent_intensity_rescale:
            m.set_model_sampler_post_cfg_function(rescale_post_cfg)
        return (m, args_str, )

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

    CATEGORY = "model_patches/automatic_cfg/presets"

    def patch(self, model, hard_mode, boost):
        advcfg = advancedDynamicCFG()
        m = advcfg.patch(model,
                         skip_uncond = boost,
                         uncond_sigma_start = 15,  uncond_sigma_end = 1,
                         automatic_cfg = "hard" if hard_mode else "soft"
                         )[0]
        return (m, )

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

    CATEGORY = "model_patches/automatic_cfg/presets"

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
                                "subtract_latent_mean_sigma_start": ("FLOAT", {"default": 15,  "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.1}),
                                "subtract_latent_mean_sigma_end":   ("FLOAT", {"default": 7.5, "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.1}),
                                "latent_intensity_rescale"     : ("BOOLEAN", {"default": True}),
                                "latent_intensity_rescale_method" : (["soft","hard","range"], {"default": "hard"},),
                                "latent_intensity_rescale_cfg" : ("FLOAT", {"default": 7.6,  "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.1}),
                                "latent_intensity_rescale_sigma_start": ("FLOAT", {"default": 15,  "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.1}),
                                "latent_intensity_rescale_sigma_end":   ("FLOAT", {"default": 7.5, "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.1}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/automatic_cfg"

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

    CATEGORY = "model_patches/automatic_cfg/presets"

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
                                "uncond_sigma_start": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                                "uncond_sigma_end":   ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                                "fake_uncond_sigma_end": ("FLOAT", {"default": 1.0,  "min": 0.0, "max": 10000.0, "step": 0.1, "round": 0.01}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/automatic_cfg/presets"

    def patch(self, model, uncond_sigma_start, uncond_sigma_end, fake_uncond_sigma_end):
        advcfg = advancedDynamicCFG()
        m = advcfg.patch(model=model, automatic_cfg = "hard",
                         skip_uncond = True, uncond_sigma_start = uncond_sigma_start, uncond_sigma_end = uncond_sigma_end,
                         fake_uncond_sigma_end = fake_uncond_sigma_end, fake_uncond_sigma_start = 1000, fake_uncond_start=True,
                         fake_uncond_exp=True,fake_uncond_exp_normalize=True,fake_uncond_exp_method="previous_average"
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

    CATEGORY = "model_patches/automatic_cfg"

    def unpatch(self, model):
        global global_skip_uncond, maximum_sigma_to_enable_uncond, minimum_sigma_to_disable_uncond
        global_skip_uncond = False # just in case of mixup with another node
        comfy.samplers.sampling_function = original_sampling_function
        maximum_sigma_to_enable_uncond, minimum_sigma_to_disable_uncond = 1000000, 0
        print(f"Sampling function unpatched.")
        return (model, )
