import torch

class simpleDynamicCFG:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                                "model": ("MODEL",),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches"

    def patch(self, model):
        top_k = 0.25
        reference_cfg = 8
        def linear_cfg(args):
            cond = args["cond"]
            cond_scale = args["cond_scale"]
            uncond = args["uncond"]
            input_x = args["input"]
            target_intensity = 0.8/reference_cfg*cond_scale
            denoised_tmp = input_x-(uncond + reference_cfg*(cond - uncond))
            for b in range(len(denoised_tmp)):
                for c in range(len(denoised_tmp[b])):
                    channel = denoised_tmp[b][c]
                    max_values = torch.topk(channel, k=int(len(channel)*top_k), largest=True).values
                    min_values = torch.topk(channel, k=int(len(channel)*top_k), largest=False).values
                    max_val = torch.mean(max_values).item()
                    min_val = torch.mean(min_values).item()
                    denoised_range = (max_val+abs(min_val))/2
                    tmp_scale = reference_cfg*target_intensity/denoised_range
                    denoised_tmp[b][c] = uncond[b][c] + tmp_scale * (cond[b][c] - uncond[b][c])
            return denoised_tmp
        m = model.clone()
        m.set_model_sampler_cfg_function(linear_cfg, disable_cfg1_optimization=True)
        return (m, )

class simpleDynamicCFGperChannelMultiplier:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",),
                             "dynamic_intensity_channel_1": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step":0.01, "round": 0.01}),
                             "dynamic_intensity_channel_2": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step":0.01, "round": 0.01}),
                             "dynamic_intensity_channel_3": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step":0.01, "round": 0.01}),
                             "dynamic_intensity_channel_4": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step":0.01, "round": 0.01}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches"

    def patch(self, model, dynamic_intensity_channel_1,dynamic_intensity_channel_2,dynamic_intensity_channel_3,dynamic_intensity_channel_4):
        base_target = 0.41903709829879954/model.model.latent_format.scale_factor
        top_k = 0.25
        dynamic_channels = [dynamic_intensity_channel_1,dynamic_intensity_channel_2,dynamic_intensity_channel_3,dynamic_intensity_channel_4]
        def linear_cfg(args):
            cond = args["cond"]
            cond_scale = args["cond_scale"]
            intensity_target = cond_scale/8
            uncond = args["uncond"]
            input_x = args["input"]
            denoised_tmp = input_x-(uncond + 8*(cond - uncond))
            for b in range(len(denoised_tmp)):
                for c in range(len(denoised_tmp[b])):
                    channel = denoised_tmp[b][c]
                    max_values = torch.topk(channel, k=int(len(channel)*top_k), largest=True).values
                    min_values = torch.topk(channel, k=int(len(channel)*top_k), largest=False).values
                    max_val = torch.mean(max_values).item()
                    min_val = torch.mean(min_values).item()
                    denoised_range = (max_val+abs(min_val))/2
                    tmp_scale = 2*base_target*intensity_target/denoised_range*dynamic_channels[c]
                    denoised_tmp[b][c] = uncond[b][c] + tmp_scale * (cond[b][c] - uncond[b][c])
            return denoised_tmp
        m = model.clone()
        m.set_model_sampler_cfg_function(linear_cfg, disable_cfg1_optimization=True)
        return (m, )
