My own version "from scratch" of a self-rescaling CFG. It isn't much but it's honest work.

While this node is connected, this will turn your sampler's CFG scale into something else.
This methods works by rescaling the CFG at each step by evaluating the potential average min/max values. Aiming at a desired output intensity.
The base intensity has been arbitrarily chosen by me and your sampler's CFG scale will make this target vary.
I have set the "central" CFG at 8. Meaning that at 4 you will aim at half of the desired range while at 16 it will be doubled. This makes it feel slightly like the usual.

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
