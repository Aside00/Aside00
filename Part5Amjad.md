# Part 5 by Amjad Althobaiti

## The Plan

| Setting | Value |
|---|---|
| TP | **8** |
| PP | **4** |
| DP | **4** |
| Precision | **FP8** |
| Activation Checkpointing | **ON** |

---

## Memory Calculation

> **Note:** The exact layer count for the 120B model is not explicitly mentioned in the core instructions, so this calculation uses **80 layers** as the baseline reference.

$$\text{Model State} = 120\text{B} \times 12 \text{ bytes/param (FP8)} = 1440 \text{ GB}$$

$$\text{Per-GPU State} = \frac{1440}{\text{TP} \times \text{PP}} = \frac{1440}{32} = 45.0 \text{ GB}$$

$$\text{Activations} = 0.8 \times \left(\frac{80}{4}\right) \times 0.35 = 5.6 \text{ GB}$$

$$\text{Total Memory per GPU} = 45.0 \text{ GB} + 5.6 \text{ GB} = \mathbf{50.6 \text{ GB}} \le 80 \text{ GB}$$

### Why TP x PP Must Equal 32
If this setup used BF16, the model parameters alone would need $120\text{B} \times 16 \text{ bytes} = 1920 \text{ GB}$, meaning $\text{TP} \times \text{PP} \ge \frac{1920}{80} = 24$. Since each dimension must be a power of two, the smallest valid product is **32**. 

For FP8, the minimum requirement drops to 18, which still rounds up to 32 under power-of-two constraints. Thus, $\text{TP} \times \text{PP} = 32$ and $\text{DP} = 4$.

### Layer Count Compatibility Check

| Layers | FP8 + Checkpointing | BF16 + Checkpointing | BF16 (No Checkpointing) |
|---|---|---|---|
| **64** | 49.5 GB | 64.5 GB | 72.8 GB |
| **80** | 50.6 GB | 65.6 GB | 76.0 GB |
| **96** | 51.7 GB | 66.7 GB | 79.2 GB |
| **120** | 53.4 GB | 68.4 GB | **84.0 GB (OOM)** |

FP8 paired with activation checkpointing stays within memory limits across all typical layer counts. Running BF16 without checkpointing causes an Out-Of-Memory (OOM) error on deeper model architectures.

---

## MFU Calculation

Using the evaluation formula provided:

$$\text{MFU} = \frac{1}{1 + \text{tp-overhead} + \text{pipeline-bubble} + \text{dp-overhead}}$$

1. **TP Overhead:**
   $$\text{tp-overhead} = 0.03 \times (\text{TP} - 1) \times \text{link-penalty} = 0.03 \times 7 \times 1 = \mathbf{0.21}$$

2. **Pipeline Bubble:**
   $$\text{pipeline-bubble} = \frac{\text{PP} - 1}{(\text{PP} - 1) + 16} = \frac{3}{3 + 16} = \mathbf{0.15789}$$

3. **DP Overhead:**
   $$\text{dp-overhead} = 0.01 \times \log_2(\text{DP}) = 0.01 \times \log_2(4) = \mathbf{0.02}$$

4. **Total Overhead Sum:**
   $$\text{Sum} = 0.21 + 0.15789 + 0.02 = \mathbf{0.38789}$$

5. **Final MFU:**
   $$\text{MFU} = \frac{1}{1.38789} \approx \mathbf{72.1\%}$$

---

## Technical Note

To satisfy the memory bounds, $\text{TP} \times \text{PP}$ must equal 32. Smaller products trigger memory failures, while larger products lower the overall efficiency score. This leaves four potential configurations for splitting the factor of 32 (with $\text{DP} = 4$ constant):

| Configuration | TP Overhead | Pipeline Bubble | DP Overhead | MFU | Memory (FP8 + Ckpt) |
|---|---|---|---|---|---|
| **TP8 / PP4 / DP4** | **0.210** | **0.158** | **0.02** | **72.1%** | **50.6 GB** |
| TP4 / PP8 / DP4 | 0.090 | 0.304 | 0.02 | 70.7% | 47.8 GB |
| TP2 / PP16 / DP4 | 0.030 | 0.484 | 0.02 | 65.2% | 46.4 GB |
| TP1 / PP32 / DP4 | 0.000 | 0.660 | 0.02 | 59.5% | 45.7 GB |

* **TP8 / PP4 vs TP4 / PP8:** Shifting a factor of 2 from PP to TP increases tensor parallelism overhead by 0.12, but reduces the pipeline bubble penalty by 0.146. Because the bubble reduction is larger than the overhead increase, **TP8 / PP4 / DP4** yields a 1.4% higher net MFU.
* **Why TP stops at 8:** A node holds up to 8 GPUs. Setting $\text{TP} = 16$ would force inter-node communication over InfiniBand, raising the link penalty from 1 to 18. That would drive tensor overhead up to $8.1$ and drop the overall MFU to roughly 11%.
* **Why FP8 and Checkpointing are used:** Neither setting introduces a penalty in the MFU scoring formula. They provide extra memory headroom to prevent potential OOM errors without lowering efficiency.
