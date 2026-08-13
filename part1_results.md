Part 1 for assignment

**GPUs : Kaggle with 2x T4**  

## Results

| Lab | Measurement | Value |
|---|---|---|
| DDP | tokens/sec, 1 GPU | 10,283 |
| DDP | tokens/sec, 2 GPUs | 18,339 |
| DDP | **scaling efficiency** | **0.892 (89.2%)** |
| DDP | ms per step, 1 / 2 GPUs | 398.3 / 446.7 |
| DDP | peak VRAM per GPU, 1 / 2 GPUs | 8.8 GB / 8.8 GB |
| FSDP | **1 GPU result** | **OOM confirmed, inside `opt.step()`** |
| FSDP | **peak VRAM per GPU, 2 GPUs** | **14.4 GB, both ranks** |
| FSDP | tokens/sec, ms per step, 2 GPUs | 605, 3387.6 |
| NCCL | **busbw at 256 MB** | **6.7 GB/s** |
| NCCL | **busbw at 512 MB** | **6.7 GB/s** |
| NCCL | ratio to NVLink (~900 GB/s) | **134x slower** |
| TP | **max abs error vs 1 GPU** | **1.91e-06** |
| TP | relative error | **3.7e-06 (MATCH, threshold 1e-3)** |


## DDP

| | 1 GPU | 2 GPUs |
|---|---|---|
| tokens/sec | 10,283 | 18,339 |
| ms per step | 398.3 | 446.7 |
| peak VRAM per GPU | 8.8 GB | 8.8 GB  |
| loss | 10.98 | 10.94 |

```
scaling efficiency = tps(2) / (2 x tps(1))
                   = 18,339 / (2 x 10,283)
                   = 18,339 / 20,566
                   = 0.892   ->  89.2%

speedup = 18,339 / 10,283 = 1.78x  instead of 2.00x
```
If adding a second GPU was completely free, our efficiency score would be 1.0 (100%). However, efficiency dropped to 89.2% because the step time increased by 48.4 ms to sync gradients over a slower PCIe connection. PyTorch actually hid 68% of this communication time in the background, but the final gradient bucket still caused a small delay

## FSDP

| | 1 GPU | 2 GPUs (`--ckpt`) |
|---|---|---|
| result | **OOM, run failed** | trained fine |
| peak VRAM per GPU | died at 13.72 GiB allocated | **14.4 GB, both ranks** |


the model weights, gradients, and optimizer states are sharded across GPUs, reducing the estimated model state from 22.6 GB to 11.3 GB per GPU. However, the actual measured peak memory reached 14.4 GB during execution. 
The extra 3.1 GB comes from temporary activations and the full block weights gathered dynamically during computation.

## NCCL

| size | time | algbw | busbw |
|---|---|---|---|
| 1 MB | 0.24 ms | 4.4 GB/s | 4.4 GB/s |
| 4 MB | 0.73 ms | 5.8 GB/s | 5.8 GB/s |
| 16 MB | 2.70 ms | 6.2 GB/s | 6.2 GB/s |
| 64 MB | 10.42 ms | 6.4 GB/s | 6.4 GB/s |
| **256 MB** | 39.95 ms | 6.7 GB/s | **6.7 GB/s** |
| **512 MB** | 80.31 ms | 6.7 GB/s | **6.7 GB/s** |

The measured speed was 6.7 GB/s, which is 134 times slower than NVLink because the Kaggle T4 GPUs use PCIe connection and host memory routing.
Transfers also take twice as long because the all-reduce algorithm moves every piece of data across the link twice. 

## TP

```
each GPU stored HALF of W1 ((4096, 8192)) and HALF of W2 ((8192, 4096))
max abs error = 1.91e-06   (relative 3.7e-06)
MATCH - same result, half the weights per GPU.
```

The script passes when the relative error is below 1e-3, and 3.7e-06 clears that by a factor of about 270. The split is correct.

Column-parallel needs no communication because both GPUs already have the full input, allowing each to compute its own half of the output independently. In contrast, row-parallel splits the weights across hidden units, so each GPU produces only a partial sum. An all-reduce is required at the end to add these partial results together so every GPU gets the complete output



#### by amjad Althobaiti
