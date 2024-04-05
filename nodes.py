from copy import deepcopy
import comfy.samplers
import torch
import math

original_sampling_function = deepcopy(comfy.samplers.sampling_function)
minimum_sigma_to_disable_uncond = 1

def sampling_function_patched(model, x, timestep, uncond, cond, cond_scale, model_options={}, seed=None):
        if math.isclose(cond_scale, 1.0) and model_options.get("disable_cfg1_optimization", False) == False or timestep[0] <= minimum_sigma_to_disable_uncond:
            uncond_ = None
            cond_scale = 1
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

class advancedDynamicCFG:
    def __init__(self):
        self.last_cfg_ht_one = 8

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "model": ("MODEL",),
                                "center_mean_post_cfg" : ("BOOLEAN", {"default": True}),
                                "center_mean_to_sigma" : ("BOOLEAN", {"default": True}),
                                "automatic_cfg" : (["None","soft","hard","include_boost"], {"default": "hard"},),
                                "sigma_boost" : ("BOOLEAN", {"default": True}),
                                "sigma_boost_percentage": ("FLOAT", {"default": 6.86, "min": 0.0, "max": 100.0, "step": 0.01, "round": 0.01}),
                                "lerp_uncond" : ("BOOLEAN", {"default": False}),
                                "lerp_uncond_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.1, "round": 0.1}),
                                # "debug_print" : ("BOOLEAN", {"default": False}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches"

    def patch(self, model, center_mean_post_cfg, center_mean_to_sigma,
              automatic_cfg, sigma_boost, sigma_boost_percentage, lerp_uncond=False, lerp_uncond_strength=1, debug_print=False):
        
        global minimum_sigma_to_disable_uncond
        model_sampling = model.model.model_sampling
        sigmin = model_sampling.sigma(model_sampling.timestep(model_sampling.sigma_min))
        sigmax = model_sampling.sigma(model_sampling.timestep(model_sampling.sigma_max))

        if sigma_boost_percentage > 0 and sigma_boost:
            minimum_sigma_to_disable_uncond  = (sigmax - sigmin) / 100 * sigma_boost_percentage
            comfy.samplers.sampling_function = sampling_function_patched
            print(f"Sampling function patched. Trigger when sigmas are at: {round(minimum_sigma_to_disable_uncond.item(),4)}")
        else:
            comfy.samplers.sampling_function = original_sampling_function
            print(f"Sampling function unpatched.")
        
        top_k = 0.25
        reference_cfg = 8
        def linear_cfg(args):
            cond_scale = args["cond_scale"]
            input_x = args["input"]
            cond_pred = args["cond_denoised"]
            uncond_pred = args["uncond_denoised"]
            if lerp_uncond:
                uncond_pred = torch.lerp(cond_pred,uncond_pred,lerp_uncond_strength)
                # uncond_pred = uncond_pred * cond_pred.norm() / uncond_pred.norm()
            cond = input_x - cond_pred
            uncond = input_x - uncond_pred
            sigma = args["sigma"][0]

            if sigma == sigmax or cond_scale > 1:
                self.last_cfg_ht_one = cond_scale
                
            target_intensity = self.last_cfg_ht_one / 10

            if sigma_boost and cond_scale > 1:
                for b in range(len(cond)):
                    for c in range(len(cond[b])):
                        uncond[b][c] = uncond[b][c] * torch.norm(cond[b][c]) / torch.norm(uncond[b][c])
            
            if automatic_cfg == "None" or (cond_scale == 1 and automatic_cfg != "include_boost"):
                return uncond + cond_scale * (cond - uncond)
            
            if cond_scale > 1:
                denoised_tmp = input_x - (uncond + reference_cfg * (cond - uncond))
            else:
                denoised_tmp = input_x + cond_pred

            for b in range(len(denoised_tmp)):
                for c in range(len(denoised_tmp[b])):
                    channel = denoised_tmp[b][c]
                    max_values = torch.topk(channel, k=int(len(channel)*top_k), largest=True ).values
                    min_values = torch.topk(channel, k=int(len(channel)*top_k), largest=False).values
                    max_val = torch.mean(max_values).item()

                    if automatic_cfg == "soft":
                        min_val = abs(torch.mean(min_values).item())
                    elif automatic_cfg == "hard" or automatic_cfg == "include_boost":
                        min_val = torch.mean(torch.abs(min_values)).item()

                    denoised_range   = (max_val + min_val) / 2
                    scale_correction = target_intensity    / denoised_range
                    tmp_scale = reference_cfg * scale_correction
                    
                    if debug_print:
                        print(f"c{c}: {tmp_scale} / {scale_correction}")

                    if cond_scale > 1:
                        denoised_tmp[b][c] = uncond[b][c] + tmp_scale * (cond[b][c] - uncond[b][c])
                    else:
                        denoised_tmp[b][c] = scale_correction * cond[b][c]

            # The scaling has been done per channel, now we set it back to norm.
            if cond_scale == 1:
                denoised_tmp = denoised_tmp * cond.norm() / denoised_tmp.norm()

            return denoised_tmp
        
        def center_mean_latent_post_cfg(args):
            denoised = args["denoised"]
            sigma    = args["sigma"][0]
            mult     = map_sigma(sigma, sigmax, sigmin) if center_mean_to_sigma else 1
            denoised = center_latent_mean_values(denoised, False, mult)
            return denoised

        m = model.clone()
        m.set_model_sampler_cfg_function(linear_cfg, disable_cfg1_optimization=False)
        if center_mean_post_cfg:
            m.set_model_sampler_post_cfg_function(center_mean_latent_post_cfg)

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

    def patch(self, model, boost, color_balance=False):
        advcfg = advancedDynamicCFG()
        m = advcfg.patch(model,color_balance,color_balance,"hard" if boost else "soft", boost, 6.86)[0]
        return (m, )

class simpleDynamicCFGlerpUncond:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "model": ("MODEL",),
                                "boost" : ("BOOLEAN", {"default": True}),
                                "negative_strength": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step": 0.1, "round": 0.1}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches"

    def patch(self, model, boost, negative_strength):
        advcfg = advancedDynamicCFG()
        m = advcfg.patch(model, False, False, "hard", boost, 6.86, negative_strength != 1, negative_strength / 2)[0]
        return (m, )
