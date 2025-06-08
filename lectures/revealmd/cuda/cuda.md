---
title: "Multimodality"
format: revealjs
---

<style>
    .reveal .slides {
        text-align: left;
    }

    .reveal table {
        font-size: 30px;
    }

</style>


## ⚡️ Efficiency

## 🎯 **Goals**

<!-- * Understand different number formats and their trade-offs.
* Gain hands-on experience optimizing model training for speed and resource use.
* Learn to profile, analyze, and improve model implementation and data pipelines. -->

---

## ⚠️ Problems

* **GPU Memory**
* **Training Speed** (Compute/Memory/Disk bottlenecks)

---

## 💾 **GPU Memory**

* Good GPU memory size for NN training: **80GB**
* **Llama3.1 8B:** **1M** tokens is a good batch size
* 100M Batch size requires **2TB** GPU memory *(naive training realization)* 🤔

![](./llama8b_gpu_mem.png)

---

## ⏳ **Training Speed**

### 💰 **1 GPU-hour = $2**


| Model | GPU-Hours |
|-------|-----------|
| Llama3.1 8B | 1.46M    |
| Llama3.1 70B | 7.0M    |
| Llama3.1 405B | 30.84M |


---

## 1️⃣ Single-GPU Training


| Feature | Training speed | Memory usage |
|---|---|---|
| batch size | ✅ | ✅ |
| gradient accumulation | ❌ | ✅ |
| gradient checkpointing | ❌ | ✅ |
| mixed precision | ✅ | ❌✅ |
| optimizers | ✅ | ✅ |
| data preloading | ✅ | ❌ |
| torch_empty_cache_steps | ❌ | ✅ |
| torch.compile | ✅ | ❌ |
| PEFT | ❌ | ✅ |
| Efficient kernels| ✅ | ✅ |

</br>

### 📚 **Resources**:

* [**HF GPU Perf Guide**](https://huggingface.co/docs/transformers/en/perf_train_gpu_one)


---

## Batch size

* Why does it matter❓ Why large batch size saves computations❓

---

## Batch size

* Why does it matter❓ Why large batch size saves computations❓
    * 💡 GPUs are optimized for high parallelization.
    * 💡 Large batch size allows for more parallelization.
    * 💡 Small batch size requires more iterations to converge.
    * 💡 Sometimes Model Params are larger then processed data. Model params loading makes task **💾 Memory-bound** not **🧮 Compute-bound**.

---

## 🔋 Gradient Accumulation

**Idea:** Split batch into smaller chunks and accumulate gradients.


```python
for i, batch in enumerate(dataloader):
    loss = model(batch)
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## 🔋 Gradient Accumulation

**Idea:** Split batch into smaller chunks and accumulate gradients.

**HF Trainer:**
```python
training_args = TrainingArguments(
    ...
    gradient_accumulation_steps=4,
)
```


---

## 📸 Gradient Checkpointing

**Idea:** Save memory by checkpointing intermediate activations and recomputing them at backward pass.

```python
training_args = TrainingArguments(
    ...
    gradient_checkpointing=True,
)
```

❓ Why do we need to save and even recompute activations at backward pass?

---

## 🍸 Mixed Precision

**Idea:** Use lower precision for some model modules to speed up training.

[**torch.amp**](https://docs.pytorch.org/docs/stable/amp.html)
```python
with torch.autocast(device_type="cuda", dtype=torch.float16):
    loss = model(batch)
    ...
```
</br>

#### **HF Trainer:**
**fp16** / **bf16**
```python
training_args = TrainingArguments(
    ...
    fp16=True,
    # bf16=True,
)
```

---

## 🍸 Automatic Mixed Precision

[**torch.amp**](https://docs.pytorch.org/docs/stable/amp.html)


### CUDA Ops that can autocast to float16
```
__matmul__, addbmm, addmm, addmv, addr,baddbmm, bmm, chain_matmul,
multi_dot, conv1d, conv2d, conv3d, conv_transpose1d,
conv_transpose2d, conv_transpose3d, linear,
matmul, mm, mv,
... and more
```

### CUDA Ops that can autocast to float32
```
__pow__, __rdiv__, __rpow__, __rtruediv__,
acos, asin, cosh,
binary_cross_entropy_with_logits, cosine_embedding_loss,
log, log_softmax, log10, log1p, log2,
mse_loss, multilabel_margin_loss, multi_margin_loss, l1_loss, nll_loss,
norm, normalize,
... and more
```

---

## 🍸 Mixed Precision

* `float32` vs `float16` vs `bfloat16` vs `int8`
* Trade-offs: Precision, speed, hardware compatibility

![](tf32-bf16-fp16-fp32.png)

---

## 🍸 Mixed Precision


❗️ **Note:**
* [bfloat16 issues](https://github.com/huggingface/transformers/issues/25420#issuecomment-1775317535) - **bf16** works worse than **fp16** for Llama3.1 models


---

## 🍸 Mixed Precision

### Task 🤔

> There is a set of floating-point numbers. We need to calculate the sum of these values.

### Naive approach

```python
sum = 0
for value in values:
    sum += value
```

### Any Problems❓

---

## 🍸 Mixed Precision

### Task 🤔

### Sort Values

```python
sum = 0
for value in sorted(values):
    sum += value
```

---

## 🍸 Mixed Precision

### Task 🤔

### Maintain Sorted Order

```python
import bisect

sum_keep_sorted_optim = 0
sorted_floats_optim = float_array.tolist()
for _ in tqdm(range(len(sorted_floats_optim) - 1)):
    sum_least_elements = sorted_floats_optim.pop(0) + sorted_floats_optim.pop(0)
    bisect.insort_left(sorted_floats_optim, sum_least_elements)
    if len(sorted_floats_optim) == 1:
        sum_keep_sorted_optim = sorted_floats_optim[0]
```

---

## 📈 Optimizers

* Adafactor - memory efficient optimizer
* 8-bit Adam - Keeps Quantized Weights in memory

---

## 📦 DataLoaders

* `num_workers`
    * Disk IO bottlenecks
* `prefetch_factor`
* `Data collators`
* Dataset caching and lazy loading
* Memory pinning

---

## ∅torch_empty_cache_steps

* 💾 Saves memory by clearing CUDA cache
* ❗️ Slows down training (up to 2x)
* ❓ Why it slows down?

```
training_args = TrainingArguments(
    ...
    torch_empty_cache_steps=4,
)
```

---

## ⚒️ torch.compile

**JIT** - Just-In-Time compilation

**How it works:**
* 💡 Python-level tracing mechanism (TorchDynamo).
* 💡 Compiles captured operations to **fused CUDA kernels**
* 💡 Caches compiled graphs
* 💡 Reuses compiled graphs
    * If input shapes/types are the same


---

## ⚒️ torch.compile

**Pros:**
* ✅ Reduces GPU memory usage (why❓)
* ✅ Speeds up training

**Cons:**
* ❗️ Graph compilation - first steps could be slow
* ❗️ Graph recomputation if input shape changes

---

## ⚒️ torch.compile


**[torch.compile](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)**
```python
model = torch.compile(model)
```

**HF Trainer:**
```python
training_args = TrainingArguments(
    ...
    torch_compile=True,
)
```

---

## ⚒️ torch.compile

<br/>
<br/>

### Resources:

* [Look Ma, No Bubbles](https://t.me/quant_prune_distill/489) - single-kernel GPT
* [torch.compile docs](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)


---

## 🔖 PEFT

* [Soft Prompts](https://huggingface.co/docs/peft/conceptual_guides/prompting)
* [LoRA](https://huggingface.co/docs/peft/conceptual_guides/adapter)


---

## 🔖 [PEFT: Soft Prompts](https://huggingface.co/docs/peft/conceptual_guides/prompting)

| Feature                         | Prompt Tuning         | Prefix Tuning               | P-Tuning                             |
| ------------------------------- | --------------------- | --------------------------- | -------------------------------------- |
| Affects All Transformer Layers? | ❌ No                  | ✅ Yes                       | ✅ Yes                                  |
| Learnable parameters type       | Soft token embeddings | MLP over soft embeddings | MLP/LSTM overt Soft Tokens             |

---

## 🔖 [PEFT: Soft Prompts](https://huggingface.co/docs/peft/conceptual_guides/prompting)

* What is the difference between Soft Prompt and Hard Prompt❓
* Why do we need MLP or LSTM over Soft Tokens❓

---

## 🔖 [PEFT: LoRA](https://huggingface.co/docs/peft/conceptual_guides/adapter)

**Idea:**
* Add trainable shift for MLPs output
* Trainable params should be small

![](lora_animated.gif)


---

## 🛑 Implementation Efficiency

> The first rule of optimization: **<span style="color:red;">Don't do it</span>**



---

## 🛑 Implementation Efficiency

> The first rule of optimization: **<span style="color:red;">Don't do it</span>**

<br/>

* Are you sure your task was never solved before❓

---

## 🛑 Implementation Efficiency

<br/>

Low-level optimizations checklist:

* ✅ Write dummy PyTorch implementation
* ✅ Use `torch.compile`
* ✅ Ask ChatGPT to optimize it
* 🛑 Use Triton
* 🛑 Use C++/CUDA

---

## 🛑 Implementation Efficiency

<br/>

Still think you need to write C++/CUDA code❓

---

## 🛑 Implementation Efficiency

<br/>

![](./debugging_segfaults.jpg)

---

## 🛑 Implementation Efficiency

<br/>

### Topics:

* CUDA Async Nature
* CUDA Architecture
* CUDA Memory Hierarchy
* CUDA Kernel Life Cycle
* CUDA Kernel fusion
* CUDA kernels with Triton
* CUDA kernels with C++/CUDA

---

## Cuda Async Nature

![async_cuda.drawio](async_cuda.drawio.png)

<br/>

**Resources:**
* [Asynchronous Execution (Torch Docs)](https://docs.pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution)

---

## Cuda Async Nature

❗️ This code measures only CPU time for kernel launch
```python
start_time = time.time()
module.forward(x)
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")

```

---

## Cuda Async Nature

❗️ use `torch.cuda.synchronize()`
```python
start_time = time.time()
module.forward(x)
torch.cuda.synchronize() # ❗️NEW LINE❗️
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")

```

---

## Cuda Async Nature

<br/>

* When do we need to run async streams❓
* Could we benefit from async streams in case of parallel compute-bound tasks that utilizes all GPU cores❓

---

## CUDA Architecture



---

## CUDA Memory Hierarchy

---


## CUDA Kernel Life Cycle

---

## CUDA Kernel fusion

---

## CUDA kernels with Triton

---

## CUDA kernels with C++/CUDA

---


---

## Outline

* TODO

---

## 💬 Wrap-Up & Discussion

### Key Questions:

* Which optimization gave the biggest improvement?
* When is quantization “worth it”?
* Where is the “easy win” vs “hard engineering”?

### Deliverables (Optional):

* Mini-report or notebook summarizing profiling results and optimizations.
* Group discussion on what trade-offs made the biggest impact.

---


## 😢 Out of scope:

* **Extreme low-bit quantization**
    * [**AWQ**](https://huggingface.co/docs/transformers/main/quantization/awq)
    * [**Quantization-aware training**](https://pytorch.org/blog/quantization-aware-training/)

<br/>
<br/>

* **Distributed training**
    * [**HF Distributed Training**](https://huggingface.co/docs/optimum/intel/neural_compressor/distributed_training)
    * [**HF Perf Training Many GPUs**](https://huggingface.co/docs/transformers/main/en/perf_train_gpu_many)
    * [**Ultrascale Playbook**](https://huggingface.co/spaces/nanotron/ultrascale-playbook)

<br/>
<br/>

* **Pytorch internals**
    * [**Pytorch Internals**](http://blog.ezyang.com/2019/05/pytorch-internals/)
    * [**A Tour of Pytorch Internals**](https://pytorch.org/blog/a-tour-of-pytorch-internals-1/)



