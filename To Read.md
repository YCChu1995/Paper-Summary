1. [2506.02153](https://arxiv.org/abs/2506.02153) Small Language Models are the Future of Agentic AI
2. [2410.06205](https://arxiv.org/abs/2410.06205) Why RoPE is useful?

### 1. FT Instability from BERT to T5 & GPT
The high generalization variance leads to fine-tuning Instability in BERT, from the read paper in file, [ON THE STABILITY OF FINE-TUNING BERT: MISCONCEPTIONS, EXPLANATIONS, AND STRONG BASELINES](https://github.com/YCChu1995/Paper-Summary/blob/main/2006_On%20the%20Stability%20of%20Fine-tuning%20BERT%20-%20Misconceptions%2C%20Explanations%2C%20and%20Strong%20Baselines.md)<br>
The high generalization variance is also found in T5 and GPT.
- [2301.09820](https://arxiv.org/abs/2301.09820) A Stability Analysis of Fine-Tuning a Pre-Trained Model 
- [2302.07778](https://arxiv.org/abs/2302.07778) Measuring the Instability of Fine-Tuning

### 2. LoRA should tuning Q + V over Q + K + V 
https://arxiv.org/abs/2410.02247

### 3. High-layer: task-specific semantics and objectives ; Low-layer: general linguistic and syntactic representations
> Low-layer: learn general linguistic and syntactic representations<br>
> High-layer: specialize to task-specific semantics and objectives &rarr; fine-tuning alters theses more.

- [1911.03090] “What Would Elsa Do? Freezing Layers During Transformer Fine-Tuning”<br>
    - The authors show that fine-tuning just the top ~25% of layers in BERT or RoBERTa can achieve ~90% of full fine-tuning performance across a variety of NLP tasks.
- [1908.05620] “Visualizing and Understanding the Effectiveness of BERT”<br>
    - They performed layer rollback experiments—resetting different groups of fine-tuned layers back to their pre-trained values.
    - Rollbacking higher layers leads to large performance drops; rollbacking lower or middle layers has little to no effect, sometimes even improving generalization.
    - This supports the view that higher layers encode task-specific information, while lower layers remain more transferable.
    - Further tuning of lower layers often leads to diminishing returns or even performance degradation.
- [2004.14448] "What Happens to BERT Embeddings During Fine-tuning?"
    - Using probing classifiers and representational similarity analysis, they find that fine-tuning predominantly alters higher layers, with lower layers largely unchanged or preserving linguistic structure.
    - Variation depends on the task (e.g. dependency parsing affects more layers, QA tasks like SQuAD involve fewer).
- [2105.15179] "How transfer learning impacts linguistic knowledge in deep NLP models?"
    - Investigating linguistic probing tasks on BERT/RoBERTa/XLNet, they conclude: core linguistic knowledge shifts towards lower layers after fine-tuning, while higher layers become more task-specific.

### 4. Scaling Laws
- [1812.06162] An Empirical Model of Large-Batch Training
- [2001.08361] Scaling Laws for Neural Language Models
- [2203.15556] Training Compute-Optimal Large Language Models

### 4. Monosemanticity
- [2309](https://arxiv.org/abs/2309.08600) Sparse Autoencoders Find Highly Interpretable Features in Language Models
- [2408](https://arxiv.org/abs/2408.05147) Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2
- [2405](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet
- [2501](https://arxiv.org/abs/2501.17148) AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders
- [2506](https://arxiv.org/abs/2506.19382) Measuring and Guiding Monosemanticity

- [2404](https://arxiv.org/abs/2404.16014) Improving Dictionary Learning with Gated Sparse Autoencoders
- [2405](https://www.lesswrong.com/posts/YiGs8qJ8aNBgwt2YN/improving-sae-s-by-sqrt-ing-l1-and-removing-lowest) Improving SAE's by Sqrt()-ing L1 & Removing Lowest Activating Features

### 5. Model Steering
- [2308](https://arxiv.org/abs/2308.10248) Steering Language Models With Activation Engineering
- [2406](https://arxiv.org/abs/2406.00045) **Personalized Steering** Personalized Steering of Large Language Models: Versatile Steering Vectors Through Bi-directional Preference Optimization
- [2501](https://arxiv.org/abs/2501.09929) **Contrastive Activation Steering** Interpretable Steering of Large Language Models with Feature Guided Activation Additions
- [2505](https://arxiv.org/abs/2505.06699) **Reference Model Steering** Model Steering: Learning with a Reference Model Improves Generalization Bounds and Scaling Laws
- [2511](https://arxiv.org/abs/2511.05408) **Weight-space Steering** Steering Language Models with Weight Arithmetic
