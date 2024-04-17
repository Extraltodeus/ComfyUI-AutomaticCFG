from copy import deepcopy
import comfy.samplers
import torch
import math

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
                    "cond_denoised": cond_pred, "uncond_denoised": uncond_pred, "model": model, "model_options": model_options}
            cfg_result = x - model_options["sampler_cfg_function"](args)
        else:
            cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale

        for fn in model_options.get("sampler_post_cfg_function", []):
            args = {"denoised": cfg_result, "cond": cond, "uncond": uncond, "model": model, "uncond_denoised": uncond_pred, "cond_denoised": cond_pred,
                    "sigma": timestep, "model_options": model_options, "input": x}
            cfg_result = fn(args)

        return cfg_result

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
        min_val = torch.mean(torch.abs(min_values)).item() if (measure == "hard" or measure == "range") else abs(torch.mean(min_values).item())
        denoised_range = (max_val + min_val) / 2
        chans.append(denoised_range)
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

def check_skip(sigma, high_sigma_threshold, low_sigma_threshold):
    return sigma > high_sigma_threshold or sigma < low_sigma_threshold

class advancedDynamicCFG:
    def __init__(self):
        self.last_cfg_ht_one = 8

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "model": ("MODEL",),

                                "automatic_cfg" : (["None","soft","hard","range"], {"default": "hard"},),

                                "skip_uncond" : ("BOOLEAN", {"default": True}),
                                "uncond_sigma_start": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 0.01, "round": 0.01}),
                                "uncond_sigma_end":   ("FLOAT", {"default": 6.86, "min": 0.0, "max": 100.0, "step": 0.01, "round": 0.01}),

                                "lerp_uncond" : ("BOOLEAN", {"default": False}),
                                "lerp_uncond_strength":    ("FLOAT", {"default": 2, "min": 0.0, "max": 10.0, "step": 0.1, "round": 0.1}),
                                "lerp_uncond_sigma_start": ("FLOAT", {"default": 100,  "min": 0.0, "max": 100.0, "step": 0.01, "round": 0.01}),
                                "lerp_uncond_sigma_end":   ("FLOAT", {"default": 6.86, "min": 0.0, "max": 100.0, "step": 0.01, "round": 0.01}),

                                "subtract_latent_mean" : ("BOOLEAN", {"default": False}),
                                "subtract_latent_mean_sigma_start": ("FLOAT", {"default": 100,  "min": 0.0, "max": 100.0, "step": 0.01, "round": 0.01}),
                                "subtract_latent_mean_sigma_end":   ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 0.01, "round": 0.01}),

                                "latent_intensity_rescale"     : ("BOOLEAN", {"default": True}),
                                "latent_intensity_rescale_method" : (["soft","hard","range"], {"default": "hard"},),
                                "latent_intensity_rescale_cfg" : ("FLOAT", {"default": 7.6,  "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                                "latent_intensity_rescale_sigma_start": ("FLOAT", {"default": 100,  "min": 0.0, "max": 100.0, "step": 0.01, "round": 0.01}),
                                "latent_intensity_rescale_sigma_end":   ("FLOAT", {"default": 50, "min": 0.0, "max": 100.0, "step": 0.01, "round": 0.01}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches"

    def patch(self, model, automatic_cfg = "None",
              skip_uncond = False, uncond_sigma_start = 50,  uncond_sigma_end = 6.86,
              lerp_uncond = False,    lerp_uncond_strength = 1, lerp_uncond_sigma_start = 100, lerp_uncond_sigma_end = 6.86,
              subtract_latent_mean     = False,   subtract_latent_mean_sigma_start      = 100, subtract_latent_mean_sigma_end     = 99,
              latent_intensity_rescale = False,   latent_intensity_rescale_sigma_start  = 100, latent_intensity_rescale_sigma_end = 50,
              latent_intensity_rescale_cfg = 8, latent_intensity_rescale_method = "hard",
              ignore_pre_cfg_func = False):
        
        global minimum_sigma_to_disable_uncond, maximum_sigma_to_enable_uncond, global_skip_uncond
        sigmin, sigmax = get_sigmin_sigmax(model)
        maximum_sigma_to_enable_uncond, minimum_sigma_to_disable_uncond = get_sigmas_start_end(sigmin, sigmax, uncond_sigma_start, uncond_sigma_end)
        lerp_start, lerp_end          = get_sigmas_start_end(sigmin, sigmax, lerp_uncond_sigma_start, lerp_uncond_sigma_end)
        subtract_start, subtract_end  = get_sigmas_start_end(sigmin, sigmax, subtract_latent_mean_sigma_start, subtract_latent_mean_sigma_end)
        rescale_start, rescale_end    = get_sigmas_start_end(sigmin, sigmax, latent_intensity_rescale_sigma_start, latent_intensity_rescale_sigma_end)

        if skip_uncond:
            global_skip_uncond = skip_uncond
            comfy.samplers.sampling_function = sampling_function_patched
            print(f"Sampling function patched. Uncond enabled from {round(maximum_sigma_to_enable_uncond.item(),2)} to {round(minimum_sigma_to_disable_uncond.item(),2)}")
        elif not ignore_pre_cfg_func:
            global_skip_uncond = skip_uncond # just in case of mixup with another node
            comfy.samplers.sampling_function = original_sampling_function
            print(f"Sampling function unpatched.")
        
        top_k = 0.25
        reference_cfg = 8
        def automatic_cfg(args):
            cond_scale = args["cond_scale"]
            input_x = args["input"]
            cond_pred = args["cond_denoised"]
            uncond_pred = args["uncond_denoised"]
            sigma = args["sigma"][0]

            if sigma == sigmax or cond_scale > 1:
                self.last_cfg_ht_one = cond_scale
            target_intensity = self.last_cfg_ht_one / 10

            if (check_skip(sigma, maximum_sigma_to_enable_uncond, minimum_sigma_to_disable_uncond) and skip_uncond) or cond_scale == 1:
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
            m.set_model_sampler_cfg_function(automatic_cfg, disable_cfg1_optimization = False)
        if subtract_latent_mean:
            m.set_model_sampler_post_cfg_function(center_mean_latent_post_cfg)
        if latent_intensity_rescale:
            m.set_model_sampler_post_cfg_function(rescale_post_cfg)
        return (m, )

class simpleDynamicCFG:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "model": ("MODEL",),
                                "boost" : ("BOOLEAN", {"default": True}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches"

    def patch(self, model, boost):
        advcfg = advancedDynamicCFG()
        m = advcfg.patch(model,
                         skip_uncond = boost,
                         uncond_sigma_start = 100,  uncond_sigma_end = 6.86,
                         automatic_cfg = "hard" if boost else "soft"
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

    CATEGORY = "model_patches"

    def patch(self, model, boost, negative_strength):
        advcfg = advancedDynamicCFG()
        m = advcfg.patch(model=model,
                         automatic_cfg="hard", skip_uncond=boost,
                         uncond_sigma_start = 100, uncond_sigma_end = 6.86,
                         lerp_uncond=negative_strength != 1, lerp_uncond_strength=negative_strength,
                         lerp_uncond_sigma_start = 100, lerp_uncond_sigma_end = 6.86
                         )[0]
        return (m, )

class postCFGrescaleOnly:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "model": ("MODEL",),
                                "subtract_latent_mean" : ("BOOLEAN", {"default": True}),
                                "subtract_latent_mean_sigma_start": ("FLOAT", {"default": 100,  "min": 0.0, "max": 100.0, "step": 0.01, "round": 0.01}),
                                "subtract_latent_mean_sigma_end":   ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 0.01, "round": 0.01}),
                                "latent_intensity_rescale"     : ("BOOLEAN", {"default": True}),
                                "latent_intensity_rescale_method" : (["soft","hard","range"], {"default": "hard"},),
                                "latent_intensity_rescale_cfg" : ("FLOAT", {"default": 7.6,  "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                                "latent_intensity_rescale_sigma_start": ("FLOAT", {"default": 100,  "min": 0.0, "max": 100.0, "step": 0.01, "round": 0.01}),
                                "latent_intensity_rescale_sigma_end":   ("FLOAT", {"default": 50, "min": 0.0, "max": 100.0, "step": 0.01, "round": 0.01}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches"

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
