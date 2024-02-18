My own version "from scratch" of a self-rescaling CFG. It ain't much but it's honest work.

# In short:

Quality > prompt following (but somehow it also feels like it follows the prompt more so... I'll let you decide)

# Usage:

![image](https://github.com/Extraltodeus/ComfyUI-AutomaticCFG/assets/15731540/7dadbec5-f7bf-4439-883e-bdd265c889e0)

That's it!

# Just a note:

Your CFG won't be your CFG anymore. It is turned into a way to guide the CFG/final intensity/brightness/saturation. So don't hesitate to change your habits while trying!

# The rest of the explaination:

While this node is connected, this will turn your sampler's CFG scale into something else.
This methods works by rescaling the CFG at each step by evaluating the potential average min/max values. Aiming at a desired output intensity (by intensity I mean overall brightness/saturation/sharpness).
The base intensity has been arbitrarily chosen by me and your sampler's CFG scale will make this target vary.
I have set the "central" CFG at 8. Meaning that at 4 you will aim at half of the desired range while at 16 it will be doubled. This makes it feel slightly like the usual when you're around the normal values.

The CFG behavior during the sampling being automatically set for each channel makes it behave differently and therefores gives different outputs than the usual.
From my observations by printing the results while testing, it seems to be going from around 16 at the beginning, to something like 4 near the middle and ends up near ~7. 
These values might have changed since I've done a thousand tests with different ways but that's to give you an idea, it's just me eyeballing the CLI's output.

I use the upper and lower 25% topk mean value as a reference to have some margin of manoeuver.

It makes the sampling generate overall better quality images. I get much less if not any artifacts anymore and my more creative prompts also tends to give more random, in a good way, different results.

I attribute this more random yet positive behavior to the fact that it seems to be starting high and then since it becomes lower, it self-corrects and improvise, taking advantage of the sampling process a lot more.

It is dead simple to use and made sampling more fun from my perspective :)

You will find it in the model_patches category.

TLDR: set your CFG at 8 to try it. No burned images and artifacts anymore. CFG is also a bit more sensitive because it's a proportion around 8.

Low scale like 4 also gives really nice results since your CFG is not the CFG anymore.


Also in general even with relatively low settings it seems to improve the quality.

It really helps to make the little detail fall into place:

![newspaperofmadness](https://github.com/Extraltodeus/ComfyUI-AutomaticCFG/assets/15731540/0b041042-dbb5-4ed7-a81f-add6e2093e02)


Here too even with low settings, 14 steps/dpmpp2/karras/pseudo CFG at 6.5 on a normal SDXL model:
![dpmpp_2m_karras_14steps_cfg6 5_](https://github.com/Extraltodeus/ComfyUI-AutomaticCFG/assets/15731540/4a7f47cf-f1c1-433a-8fa5-2c61c4c6f9c0)

