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

### 4. LoRA follow-ups
- [2303.10512] AdaLoRA (Hard)
- [2305.14314] QLoRA
- [2309.14717] QA‑LoRA
- [2310.11454] VeRA
- [2311.09578] Tied‑LoRA
- [2402.09353] DoRA (Hard)
