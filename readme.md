# Jimenez 2014 Bloom

![](readme/bloom.png)

A [wgpu] implementation of the bloom method presented in [Next Generation Post Processing in Call of Duty: Advanced Warfare].

Additionally, the [bevy bloom PR] and the [Unity implemation](https://github.com/Unity-Technologies/Graphics/blob/master/com.unity.postprocessing/PostProcessing/Shaders/Builtins/Bloom.shader) were very helpful in sorting out a few details.

Uses compute shaders written in [rust-gpu].

I'm planning on porting this to Vulkan to see if I can optimise things further without the limitations of [wgpu].

[wgpu]: https://github.com/gfx-rs/wgpu
[Next Generation Post Processing in Call of Duty: Advanced Warfare]: http://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare
[rust-gpu]: https://github.com/EmbarkStudios/rust-gpu
[bevy bloom PR]: https://github.com/bevyengine/bevy/pull/2876
