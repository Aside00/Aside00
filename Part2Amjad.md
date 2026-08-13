# Part 2
By:Amjad Althobaiti

## Scenario 1: Dense 34B Model on 32 GPUs (4 Nodes x 8 GPUs x 80 GB)

### The plan
*   **TP = 4, PP = 2, DP = 4**
*   **Activation Checkpointing:** on
*   **Precision:** BF16
*    4 * 2 * 4 = 32 total GPUs.

### Memory Calculations
**I asume the model has 60 layers I use the standard 34B architecture (search from web)**

*   **Model State:** 34B * 16 bytes = 544 GB
*   **Per-GPU State:** 544 / (4 * 2) = 68.0 GB
*   **Activations:** 0.8 * (60 / 2) * 0.35 = 8.4 GB
*   **Total Memory:** 68.0 + 8.4 = **76.4 GB** <= 80 GB (**Fits**)

The  TP * PP must be at least 8 because 544 GB / 80 GB = 6.8.  

Using TP=4 and PP=2 is better than TP=8 and PP=1. With TP=8, PP=1, all 60 layers stay on one stage, spiking activation memory to 16.8 GB and pushing total memory to 84.8 GB and splitting the factor of 8 as 4 * 2 halves the activation memory because activations are divided by PP.

### Hardware Mapping
Each node has 8 GPUs, TP=4 * PP=2 = 8 GPUs. will means one full model copy fits inside a single node**:

*   **TP = 4 (Inside Node, NVLink):** GPUs 0-3 are Stage 0; GPUs 4-7 are Stage 1.
*   **PP = 2 (Inside Node, NVLink):** The pipeline transfer also stays on fast NVLink.
*   **DP = 4 (Across Nodes, InfiniBand):** Each of the 4 nodes acts as one data-parallel replica.

### Communication
*   **TP = 4:** All-reduce on activations -> *NVLink (Inside Node)* -> Twice per transformer layer.
*   **PP = 2:** Point-to-point transfers -> *NVLink (Inside Node)* -> Once per micro-batch per boundary.
*   **DP = 4:** Gradient all-reduce -> *InfiniBand (Across Nodes)* -> Once per step with backward pass.

---

## Scenario 2: MoE 600B Model (8 Active Experts) on 128 GPUs (16 Nodes x 8 GPUs x 80 GB) 

### The Plan
*   **TP = 8, PP = 8, DP = 2, EP = 16**
*   **Precision:** FP8 
*   **Activation Checkpointing:** on
*    TP * PP * DP = 8 * 8 * 2 = 128 total GPUs.

### Memory Calculations
**I asume 10% dense parameters (60B) and  90% expert parameters (540B), 64 total experts, 64 layers**

I chouse  FP8 because per GPU here Have 80 GB limit and FP8 help memory down to 64.1 GB, leaving enough free space for token communication.
*   **Dense Params:** (60B * 12) / (8 * 8) = 11.2 GB
*   **Expert Params:** (540B * 12) / (8 * 16) = 50.6 GB
*   **Activations:** 0.8 * (64 / 8) * 0.35 = 2.2 GB
*   **Total Memory:** 11.2 + 50.6 + 2.2 = **64.1 GB** <= 80 GB (**Fits**)


### Hardware Mapping
*   **TP = 8:** Inside 1 node over NVLink.
*   **PP = 8:** Across nodes over InfiniBand (transfers small activation tensors so InfiniBand it good ).
*   **DP = 2:** Across the 2 nodes that form a single pipeline stage.
*   **EP = 16:** Spans 2 nodes (16 GPUs hold the 64 experts for that stage, meaning 4 experts per GPU).

### MoE Communication 
MoE layers require ** two all-to-all communication calls**:
1.  **Dispatch:** Sending tokens to the GPUs holding their selected top-8 experts.
2.  **Combine:** Returning the computed outputs back to the original token GPU.

Because EP=16 spans across 2 nodes, part of this transfer happens over InfiniBand and to keep performance high:
*   Keep EP groups small 16 GPUs across 2 nodes is much faster than spreading over 8 nodes.
*   Limit routing so tokens do not fan out to too many distant nodes.
*   Overlap dispatch/combine transfers with the computation of previous micro-batches.

---

## Scenario 3: Dense 13B Model with 1M Context Window on 16 GPUs (2 Nodes x 8 GPUs x 80 GB)

### The Plan
*   **TP = 8 , CP = 2, PP = 1, DP = 1**
*   ** Sequence Parallelism** on
*   **Precision:** BF16
*   **Activation Checkpointing:** on
*    TP * CP = 8 * 2 = 16 total GPUs.

### Memory Calculations
*I asume  40 layers, hidden size 5120 I use the standard 13B architecture (search from web).*

*   **Model Weights:** (13B * 16 bytes) / 8 (TP) = 26.0 GB per GPU
*   **Raw Activations:** 1,000,000 tokens * 5120 * 2 bytes * 40 layers = 409.6 GB

With **CP = 2** and **TP = 8 (with Sequence Parallelism)**:
*   **Activations per GPU:** 409.6 GB / (2 * 8) = 25.6 GB
*   **Total Memory:** 26.0 + 25.6 = **51.6 GB** <= 80 GB (**Fits**)

### Hardware Mapping
*   **TP = 8 (+ SP):** Inside each node over NVLink (Node 0 is TP Group 0, Node 1 is TP Group 1).
*   **CP = 2:** Across the 2 nodes over InfiniBand (Node 0 handles tokens 1-500k; Node 1 handles 500k-1M).

### Why CP is Mandatory Here
1.  **TP does not split sequence length:** TP splits the hidden dimension (width), not sequence length. At 1M tokens, activation memory explodes regardless of TP.
2.  **PP hurts performance here:** Pipeline parallelism requires many micro-batches to be efficient. With a 1M context window, the batch size is usually just 1 sequence, causing massive idle time (pipeline bubbles).
3.  **CP scales well over InfiniBand:** In Context Parallelism, compute scales quadratically while K/V communication scales linearly. For long contexts, there is plenty of computation time to hide the K/V ring transfers in the background over InfiniBand.
