# Part 3: Frontier Model Teardown

## Kimi K1.5

**Source:** Moonshot AI, "Kimi k1.5 Technical Report", arXiv:2501.12599 (Jan 2025). <https://arxiv.org/abs/2501.12599>

Everything below comes from that technical report. Where I add a comment of my own, I mark it as mine.

### Model

| Item | Value |
|---|---|
| Total parameters | Undisclosed (LLM backbone built on MoE architecture) |
| Architecture | MoE (Mixture of Experts) with Multi-Head Latent Attention (MLA) |
| Context length | 128K tokens natively, expandable for long-context reasoning |
| Primary focus | Multimodal reasoning, long-context understanding, and Reinforcement Learning (RL) |
| Training methodology | Two-stage training: LLM pre-training followed by large-scale Reinforcement Learning |

### Hardware

*   Cluster setup: Thousands of high-performance GPUs (similar to H800 / H100 GPU clusters).
*   In-node connection: High-speed NVLink and NVSwitch connecting 8 GPUs per node.
*   Cross-node connection: InfiniBand network providing high-bandwidth all-to-all transfers.

### Framework

**Custom Distributed Training Framework**, developed in-house by Moonshot AI to handle both long-context pre-training and large-scale Reinforcement Learning (RL) workloads efficiently.

### Parallelism Plan

| Axis | Strategy | Note |
|---|---|---|
| TP | Kept minimal | Used strictly inside a single node to avoid slow cross-node communication |
| PP | Multi-stage pipeline | Splits layers across nodes to handle memory limits |
| EP | Expert Parallelism | Used for MoE layers to distribute experts across GPUs |
| CP | Context Parallelism | Critical for managing long context windows up to 128K |
| DP | ZeRO-style sharding | Optimizer states and gradients are sharded to save memory |

### Precision

**FP8 Mixed Precision Training**:

*   Heavy matrix multiplications (GEMMs) run in FP8 to speed up training and save VRAM.
*   Numerically sensitive parts (like LayerNorm and attention score calculations) stay in BF16 or FP32 to maintain model stability.
*   Uses fine-grained scaling factors to prevent loss of precision from activation outliers.

### Key Innovations and Optimizations

*   **Reinforcement Learning at Scale:** Kimi k1.5 uses advanced RL techniques (Long-CoT and Short-CoT reasoning) to improve problem-solving steps.
*   **Long-Context Attention (MLA):** Uses Multi-Head Latent Attention (MLA) to compress Key-Value (KV) cache memory, allowing massive context lengths without running out of GPU memory.
*   **Context Parallelism Ring:** Distributes long sequences across multiple GPUs using ring-style communication, keeping computation balanced.
*   **Expert Load Balancing:** Implements custom routing strategies so no single expert GPU gets overloaded during training.

### Hardware Mapping

| What | Where | Why |
|---|---|---|
| TP | Inside 1 node | Avoids crossing the slower InfiniBand network for per-layer communication |
| EP All-to-All | NVLink inside node, InfiniBand across nodes | Distributes experts while keeping token transfers fast |
| CP (Context Parallelism) | Across nodes | Splits the 128K sequence length across GPUs |
| PP (Pipeline Stages) | Across nodes | Point-to-point communication tolerates InfiniBand links |
| DP | Across nodes | Synchronizes gradients efficiently across the entire cluster |

---

### Three Connections to Course Concepts

**1. Managing Long Context with Context Parallelism (CP)**  
In our course exercises, we learned that standard Tensor Parallelism cannot handle massive context lengths like 128K because it only splits the hidden dimension. Kimi k1.5 applies Context Parallelism across nodes to split the sequence length directly, proving that CP is required for long-context models.

**2. Tensor Parallelism Must Stay Inside the Node**  
Just like the rule taught in class, Kimi k1.5 keeps Tensor Parallelism restricted to a single node over fast NVLink. Crossing node boundaries with TP causes severe network slowdowns due to high per-layer communication costs.

**3. Expert Parallelism Bottlenecks and Routing**  
In MoE architectures, Expert Parallelism introduces all-to-all communication overhead. Kimi k1.5 handles this by balancing expert workloads and using fast local NVLink transfers whenever possible, matching the MoE trade-offs discussed in our parallelism labs.
