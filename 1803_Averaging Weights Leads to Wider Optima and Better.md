# Averaging Weights Leads to Wider Optima and Better
> [1803.05407](https://arxiv.org/pdf/1803.05407)<br>
> SWA (Stochastic Weight Averaging)

## Motivation 
1. Ensemble methods improve accuracy with a high inference cost.
2. Traditional training procedure only picks the best model from a training trajectory, wasting information hiding among the rest of the checkpoints.

## Summary 
Instead of picking the final weights of a model at the end of training, SWA computes a **_running average of weights_** sampled from the **_tail of the training trajectory_** (~>75%), typically from a cyclical or constant learning rate schedule.

## Tech Insights 
1. Use cyclical learning rate with SGD or Adam to **_explore different modes of well-fine-tuned models in a low-loss basin_**.
2. **_Average different modes_** of well-fine-tuned models leads to a flatter solution, which **_correlates with better generalization_**.
