# Measuring the Intrinsic Dimension of Objective Landscapes
> [1804.08838](https://arxiv.org/abs/1804.08838)<br>
> Intrinsic Dimension
<div align=center><img src="/figures/1804.08838.01.png" style="height: 150px; width: auto;"/></div>

## Summary 
1. Introducing subspace training to `measure intrinsic dimension`.
2. The measured intrinsic dimension `reveals redundancy` and `enables compression` across domain and architecture for understanding the optimization complexity of different ML problems.
3. Empirical evidence of parameter redundancy.
4. MDL perspective: `Intrinsic dimension approximates the number of parameters required to encode a model`, linking optimization to information theory.

## Tech Insights 
1. Most models have `low intrinsic dimension` and `high parameter redundancy`.
2. `Necessary parameters` (within      intrinsic dimension) bring the model to a `low-loss basin`.
3. `Redundant parameters` (ones beyond intrinsic dimension) introduce more directions in parameter space & `increase flatness of the low-loss basin` but do `NOT improve task performance` or solution efficiency.

---

## Motivation 
1. At the moment, there is a growing consensus, `"deep networks are overparameterized, many parameters are redundant"`.
2. Prior work explored compression (pruning, quantization, low-rank); theoretical aspects (MDL, generalization bound), but there is `no empirical way to measure intrinsic dimension` of the optimization landscape.

## Experiment
1. Different `architectures` usually have `different intrinsic dimensions` even for the same task.
  > In CV, CNNs are more efficient architectures than FC
2. Choosing the proper architecture for the task can have fewer intrinsic dimensions. 
  > Inverted pendulum w/ RL ~100× easier than MNIST w/ FC
<div align=center><img src="/figures/1804.08838.02.png" style="height: 200px; width: auto;"/></div>
<div align=center><img src="/figures/1804.08838.03.png" style="height: 300px; width: auto;"/></div>
