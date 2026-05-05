# QLORA: Efficient Finetuning of Quantized LLMs
> [2305.14314](https://arxiv.org/abs/2305.14314)<br>
> QLORA
<div align=center><img src="/figures/2305.14314.01.png" style="height: 200px; width: auto;"/></div>

## Summary 
- Quantize the frozen pretrained model to NF4 via DQ.
  And train LoRA adapters to `cancel out the quantization error` and `capture task-specific information`.
  > Frozen pretrained-model: Storage in **NF4**, Calculate in **BF16**.<br>
  > LoRA adapters: Both Storage and Calculate in **BF16**.
  
- LoRA
<div align=center><img src="/figures/2305.14314.02.png" style="height: 30px; width: auto;"/></div>

- QLoRA
<div align=center><img src="/figures/2305.14314.03.png" style="height: 30px; width: auto;"/></div>
<div align=center><img src="/figures/2305.14314.04.png" style="height: 30px; width: auto;"/></div>

## Tech Insights 
1. NF4 to save memory footprint.
   > Saving 11.5 bits per weight<br>
   > 16 bits [BF16] &rarr; 4 bits [NF4] + 32/64 bits [FP32/batchsize-64]
2. Double quantization on NF4 quantization constants to further save bits.
   > Saving ~11.9 bits per weight<br>
   > 16 bits [BF16] &rarr; 4 bits [NF4] + 8/64 bits [FP8/batchsize-64] + 32/(64 · 256) [FP32/(batchsize-64 · blocksize-256)]
4. Paged optimizers using unified CPU/GPU memory to avoid memory spikes during training

---

## Motivation 
1. Prior quantization methods, INT8 or INT4 techniques, work well for inference but fail during fine-tuning due to accuracy collapse.
2. Full 16-bit finetuning of massive models requires too many FLOPs.
   > (e.g. LLaMA‑65B demands >780 GB GPU RAM)
3. Even with PEFT, most memory footprint was spent on activation gradients and not the learned LoRA parameters.
   > LLaMA-7B on FLAN v2 w/ batch size = 1, w/ LoRA weights = 0.2% of the original model weights<br>
   > 4-bit base model: 5,048 MB<br>
   > LoRA input gradients: 567 MB<br>
   > LoRA parameters: 26 MB

## Chain of Thoughts
### Issue 1: Most memory footprint was NOT spent on learned LoRA parameters.

**&darr;** Quantizing the frozen pretrained model to save memory usage.

### NF4 Quantization (4-bit Normal Float) 
The author designed the NF4 from quantile quantization, since `weights in most layers of pretrained models are in the normal distribution`.
> The statement, "weights in most layers of pretrained models are in the normal distribution", is confirmed in Appendix F of the paper.

**&darr;**

### Issue 2: Outlier weights lead to significant quantization error.<br>

**&darr;** Chunking weights into a small batch can reduce outlier-oriented quantization error.

### Issue 3: Small batch-size chunking strategy leads to memory overhead from storing the quantization factor for each batch.
- In the paper, batch-size: 64 & quantization factor: FP32<br>
  &rarr; 32/64 = `0.5 bits memory overhead per weight`

**&darr;** Further quantizing quantization factors 

### Double Quantization
- The author quantize `256 block` of quantization factors in FP32 to `FP8`.<br>
  &rarr; 8/64 + 32/(64 · 256) = `0.127 bits memory overhead per weight`
