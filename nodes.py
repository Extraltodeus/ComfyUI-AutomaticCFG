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
#It would have made more sense to make the min_val absolute before the mean but I've tried that way and since it's an arbitrary goal I'll leave it like that.
#Edit 2 to explain further: fixing this makes the limit too hard and starts to visibly limit the denoising. Leaving it likes this allows a margin of manoeuver.
