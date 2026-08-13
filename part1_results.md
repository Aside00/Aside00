# Part 1 · Measurement (Scale Out: From 2 GPUs to a Datacenter)

**Status: all four labs are done.** Every number below was measured on Kaggle with 2x T4. Nothing is left to run for Part 1.

Two things came out exactly as predicted from reading the code, which is a good consistency check to mention: the param counts (255M for the DDP cells, 1.42B for the FSDP cells) and the shard shapes in the TP lab.

## Results

| Lab | Setup (from the notebook cells) | Number to report | Value |
|---|---|---|---|
| DDP | `train_ddp.py --dim 1024 --layers 12`, 1 GPU | tokens/sec | **10,283** |
| DDP | same script, 2 GPUs | tokens/sec | **18,339** |
| DDP | computed | scaling efficiency | **0.892 (89.2%)** |
| DDP | 1 GPU / 2 GPU | ms per step | **398.3 / 446.7** |
| DDP | 1 GPU / 2 GPU | peak VRAM per GPU | **8.8 GB / 8.8 GB** |
| FSDP | `train_fsdp.py --dim 2048 --layers 24`, 1 GPU | OOM confirmed? | **Yes, OOM inside `opt.step()`** |
| FSDP | same script, 2 GPUs, `--ckpt` | peak VRAM per GPU | **14.4 GB (both ranks)** |
| FSDP | 2 GPUs | tokens/sec, ms per step | **605, 3387.6** |
| NCCL | `bench_nccl.py`, 2 GPUs | busbw at 256 MB | **6.7 GB/s** |
| NCCL | same | busbw at 512 MB | **6.7 GB/s** |
| NCCL | computed | ratio to NVLink (~900 GB/s) | **134x slower** |
| TP | `tp_from_scratch.py`, 2 GPUs | max abs error vs 1 GPU | **1.91e-06** |
| TP | same | relative error | **3.7e-06 (MATCH, threshold 1e-3)** |

**Fixed facts I read directly from the notebook code (safe to write in your report):**

| Item | Value | Where it comes from |
|---|---|---|
| DDP model size | 255M params | `count_params` with dim=1024, layers=12, vocab=50304, seq=512 |
| DDP per-GPU batch | 8 (same in both runs) | `--batch` default in `train_ddp.py` |
| DDP tokens counted, 1 GPU | 122,880 (1 x 30 steps x 8 x 512) | `tokens = world * steps * batch * seq` |
| DDP tokens counted, 2 GPU | 245,760 | same formula, world=2 |
| FSDP model size | 1.42B params | `count_params` with dim=2048, layers=24 |
| FSDP full-training memory | about 23 GB at 16 bytes/param | the script prints `p*16/1e9` |
| FSDP sharded over 2 GPUs | about 11 GB/GPU before activations | the script prints `p*16/1e9/world` |
| NCCL busbw factor at world=2 | exactly 1.0, so busbw = algbw | `busbw = algbw * 2*(world-1)/world` |
| TP shapes | W1 (4096, 16384) split into two (4096, 8192); W2 (16384, 4096) split into two (8192, 4096) | `tp_from_scratch.py`, D=4096 |

## DDP

**Measured on Kaggle, 2x T4, model 255M params, per-GPU batch 8 in both runs.**

| | 1 GPU | 2 GPUs |
|---|---|---|
| tokens/sec | 10,283 | 18,339 |
| ms per step | 398.3 | 446.7 |
| peak VRAM per GPU | 8.8 GB | 8.8 GB (both ranks) |
| loss | 10.98 | 10.94 |

```
scaling efficiency = tps(2) / (2 x tps(1))
                   = 18,339 / (2 x 10,283)
                   = 18,339 / 20,566
                   = 0.892   ->  89.2%

speedup = 18,339 / 10,283 = 1.78x  instead of 2.00x
```

This lands right at the top of the 0.8 to 0.9 range the script itself predicts for PCIe.

**Explanation.**
Scaling efficiency tells you how much of the second GPU you actually got. A value of 1.0 would mean the second GPU was free, and you never see that. DDP keeps a full copy of the model on each GPU and splits only the data, so at the end of every backward pass the two GPUs must all-reduce the gradients so both copies stay identical. That all-reduce is network time, not compute time. PyTorch overlaps most of it with the backward pass by sending gradient buckets as soon as they are ready, but the last bucket cannot be hidden, and the two ranks also have to wait for each other at the barrier.

**The clearest way to see the tax is the step time, not the throughput.** Each GPU does exactly the same amount of work in both runs, because the per-GPU batch stays at 8. So if the second GPU were free, the step time would not change at all. It went from 398.3 ms to 446.7 ms, which is **48.4 ms of extra time per step, or 12.2% slower**. That 48.4 ms is the part of the gradient all-reduce that DDP could not hide behind the backward pass, plus the barrier wait.

The efficiency number is just the other side of the same fact: 398.3 / 446.7 = 0.892, the same 89.2%. That is a useful check that the measurement is consistent.

The link is the main reason the number is not higher. The two T4s on Kaggle are not connected by NVLink, so all gradient traffic goes over PCIe. The model in this cell is small (255M params at dim=1024), which makes it worse, because a small model has less compute per step to hide the communication behind. Small model plus slow link means a high communication-to-compute ratio.

**Cross-check with Lab J (this is worth putting in the report).** The gradients are FP32, so the full gradient buffer is 255M x 4 bytes = 1.018 GB. Lab J measured the link at 6.7 GB/s, so a full all-reduce of that buffer should take 1.018 / 6.7 = **152 ms**. But only 48.4 ms actually showed up in the step time. So DDP hid **103.6 ms, about 68% of the all-reduce**, behind the backward pass.

That also puts a number on what the overlap is worth:

```
no overlap at all : step = 398.3 + 152.0 = 550.3 ms  ->  efficiency 0.724
what we measured  : step = 446.7 ms                  ->  efficiency 0.892
perfect overlap   : step = 398.3 ms                  ->  efficiency 1.000
```

The bucketed gradient overlap moved the efficiency from 72% to 89%. Everything left on the table is the last bucket, which cannot start until the first layer's gradients exist.

**Two extra observations from the output:**

1. **Peak VRAM is 8.8 GB on 1 GPU and 8.8 GB on each of the 2 GPUs.** Adding a second GPU saved zero memory. This is exactly what DDP is: a full copy of the weights, gradients and optimizer states on every GPU. Only the data is split. Keep this number, because the FSDP lab is the direct contrast and this is the sentence that sets it up.
2. **Loss is about 10.9 in both runs.** The data is random tokens from a vocab of 50,304, and ln(50,304) = 10.83. So the loss sitting just above that is correct: the model cannot learn anything from random data, and it is not supposed to. These labs measure speed, not learning.

One detail worth writing down: the per-GPU batch stays at 8 in both runs, so the 2-GPU run processes twice as many tokens per step. This is weak scaling. You are measuring throughput growth at fixed per-GPU work, which is the normal way to report DDP scaling.

## FSDP

**Measured on Kaggle, 2x T4, model 1.42B params, per-GPU batch 2.**

| | 1 GPU | 2 GPUs (`--ckpt`) |
|---|---|---|
| result | **OOM, run failed** | trained fine |
| peak VRAM per GPU | died at 13.72 GiB allocated | **14.4 GB, both ranks** |
| tokens/sec | n/a | 605 |
| ms per step | n/a | 3387.6 |

**1. The OOM is confirmed, and where it happened is the interesting part.**

The run did not die during the forward pass. It got through forward and backward, then died inside `opt.step()`, on this exact line of AdamW:

```
state["exp_avg_sq"] = torch.zeros_like(...)
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 64.00 MiB.
GPU 0 has a total capacity of 14.56 GiB of which 32.81 MiB is free.
```

`exp_avg_sq` is AdamW's **second moment buffer**, the last of the four things that make up the 16 bytes per param. So the weights fit, the gradients fit, part of the first moment fit, and the optimizer state is what pushed it over the edge. That is a very clean demonstration of why optimizer states matter as much as weights.

The arithmetic lines up with the error message:

```
1.42B params x 4 bytes (FP32) = 5.66 GB per buffer
weights + gradients           = 10.54 GiB
card capacity                 = 14.56 GiB
allocated when it died        = 13.72 GiB
```

So after weights and gradients there was only about 3.2 GiB left, but AdamW needs 10.5 GiB for its two moments. It filled roughly a third of that and then ran out. Full requirement is 22.6 GB, which the script rounds to 23 GB.

Small detail that confirms the story: the failed allocation was **exactly 64.00 MiB**, and a `Linear(2048, 8192)` in FP32 is 2048 x 8192 x 4 bytes = 64 MiB exactly. That is the first MLP matrix of a Block. So it died building the optimizer state for one specific MLP layer.

**2. What got sharded: parameters, gradients, and optimizer states.**

`ShardingStrategy.FULL_SHARD` is ZeRO-3, and it splits all three. Each GPU permanently keeps only its own slice, so 22.6 GB of model state becomes about 11.3 GB per GPU. When a block is actually needed, FSDP all-gathers the full weights for just that one block, runs it, then throws the gathered copy away. In backward it reduce-scatters the gradients so each rank keeps only its shard. The `transformer_auto_wrap_policy` wraps each `Block` as its own shard unit, which is what makes "one block at a time" possible instead of gathering the whole model at once.

The 1-GPU run proves the same point from the other direction. It printed:

```
UserWarning: FSDP is switching to use NO_SHARD instead of
ShardingStrategy.FULL_SHARD since the world size is 1.
```

With one GPU there is nobody to shard with, so FSDP turned sharding off completely and behaved like plain single-GPU training. Nothing was split, and it died. Sharding needs a second GPU to be sharding at all.

**3. The peak VRAM number, and why it is higher than 11 GB.**

Measured peak is **14.4 GB per GPU**, against the 11.3 GB the script estimates for model state alone. The extra 3.1 GB is everything the estimate leaves out: the activations that survive checkpointing, the temporarily gathered full weights of the block currently running, and NCCL communication buffers.

Note this is 14.4 GB in decimal units, because the script divides by `1e9`. The card is 14.56 GiB, which is 15.63 GB in the same units. So the run peaked at about **92% of the card**. It fits, but only just.

**4. Honest notes to include in your write-up.**

1. The 2-GPU cell passes `--ckpt` and the 1-GPU cell does not, so the two runs do not have identical activation memory. The 1-GPU run still died on optimizer state, not activations, so the lesson holds. Say this rather than claiming a clean comparison.
2. On a T4 `torch.cuda.get_device_capability()[0]` is 7, so `bf16` is False and the script sets `mp = None`. **FSDP here runs in plain FP32 with no mixed precision at all**, unlike the DDP lab which uses FP16 autocast. This is why the 4 bytes per buffer above is correct, and it matters for the speed comparison below.
3. **The script's last printed line is wrong for this config.** It says "compare peak VRAM/GPU here to the ~44 GB the full model would need on one card". That 44 GB is hardcoded and refers to the script's *default* setting of dim=2560, layers=32, which is a 2.78B model. The cell you actually ran uses `--dim 2048 --layers 24`, which is 1.42B and **23 GB**, as the header line correctly printed. Use 23 GB in your report, not 44 GB.

**5. The price of FSDP: it is slow.**

DDP hit 18,339 tokens/sec. FSDP hit 605 tokens/sec, about 30 times less. Four reasons, and it is worth listing them because "FSDP is slower" on its own is not an answer:

- The model is **5.6 times bigger** (1.42B against 255M), so each step is far more compute.
- FSDP **all-gathers the weights of every block twice per step**, once in forward and once in backward, and all of that goes over PCIe. DDP only communicates gradients once, at the end.
- `--ckpt` is on, so every block's forward is **recomputed during backward**. That is roughly 30% extra compute, traded for memory.
- The DDP run used FP16 autocast and hit the T4 Tensor Cores. This FSDP run is **pure FP32**, which on a T4 is many times slower for matmuls.

That trade is the whole lesson of the lab. FSDP is not a speed feature. It is what you use when the model does not fit, and you pay for it in throughput.

## NCCL

**Measured on Kaggle, 2x T4, NCCL all-reduce, FP32 buffers.**

| size | time | algbw | busbw |
|---|---|---|---|
| 1 MB | 0.24 ms | 4.4 GB/s | 4.4 GB/s |
| 4 MB | 0.73 ms | 5.8 GB/s | 5.8 GB/s |
| 16 MB | 2.70 ms | 6.2 GB/s | 6.2 GB/s |
| 64 MB | 10.42 ms | 6.4 GB/s | 6.4 GB/s |
| **256 MB** | 39.95 ms | 6.7 GB/s | **6.7 GB/s** |
| **512 MB** | 80.31 ms | 6.7 GB/s | **6.7 GB/s** |

**The interconnect ceiling is 6.7 GB/s.**

```
NVLink on a datacenter GPU : ~900 GB/s
measured here              :    6.7 GB/s
900 / 6.7                  =  134x slower,  or 0.74% of NVLink
```

**Why busbw and algbw are identical.** The script computes `busbw = algbw * 2 * (world - 1) / world`. With `world = 2` that factor is exactly `2 * 1 / 2 = 1.0`, so the two columns must print the same value. This is not a bug, and it is worth one sentence in the report so the grader knows you understood the formula rather than copied the table. The factor only starts to matter at 4 GPUs and above.

**Why the number is so far below NVLink.** It is the wire. The two Kaggle T4s have no NVLink at all, so NCCL falls back to PCIe. A ring all-reduce also has to move each element across that link twice, once going out in the reduce-scatter phase and once coming back in the all-gather phase, so the link speed is charged twice.

For reference, PCIe Gen3 x16 (normal for a T4) is about 15.75 GB/s per direction on paper, and we measured 6.7 GB/s, which is 43% of that. Falling well short of the paper number is expected when the two cards cannot do direct peer-to-peer transfers and NCCL has to bounce the data through host memory, which means each byte crosses the PCIe bus twice more. **To confirm which case you are in, run `!nvidia-smi topo -m` in a cell.** If it prints `SYS` or `PHB` for the GPU0 to GPU1 pair, the traffic is going through the host, and that is your explanation. If it prints `PIX`, they share a switch and the gap is protocol overhead alone. I did not run this, so treat the host-memory reading as the likely explanation rather than a confirmed one.

Also note the script's own hint line quotes a PCIe 5.0 x16 peak of about 64 GB/s. **That line is hardcoded for the 2x5090 machine the labs were originally written on, not for Kaggle T4s.** Do not quote 64 GB/s in your report.

**Why the small sizes look worse than the large ones.** Fitting a straight line to the two largest points gives an asymptotic bandwidth of 6.65 GB/s and a fixed cost of roughly 0.08 ms per call. At 512 MB that fixed cost is nothing next to 80 ms of transfer. At 1 MB the transfer itself only needs 0.16 ms, so the 0.08 ms of kernel launch and synchronisation is a third of the total, and the apparent bandwidth collapses to 4.4 GB/s. This is why the assignment asks for the 256 to 512 MB numbers specifically: only the large sizes measure the wire instead of the software.

**What this number explains in the other labs.**

1. **It explains the DDP result exactly.** 1.018 GB of gradients at 6.7 GB/s is 152 ms, but DDP only lost 48.4 ms, so 68% of it was hidden behind the backward pass. The full working is in the DDP section above.
2. **It explains why FSDP was 30 times slower.** FSDP all-gathers every block's weights twice per step over this same 6.7 GB/s link, and unlike DDP's gradient all-reduce, the forward pass cannot continue until the gathered weights arrive.
3. **It is the concrete answer to Part 4 question 3, on why TP must live inside a node.** As an estimate for the DDP model (batch 8, seq 512, dim 1024, FP16), one TP all-reduce on a layer output would be 8.4 MB, and with two per layer across 12 layers that is about 200 MB per forward, so roughly 67 ms per step for forward and backward together. That is more exposed time than DDP's entire gradient all-reduce, even though it moves less than half the data. The reason is that **none of it can be overlapped**: the next operation in the layer needs the all-reduce result immediately, so the GPU just waits. DDP's collective happens at the end of backward and can be hidden. TP's cannot. On a 6.7 GB/s link, TP is unusable.

## TP

**Measured on Kaggle, 2x T4.**

```
each GPU stored HALF of W1 ((4096, 8192)) and HALF of W2 ((8192, 4096))
max abs error = 1.91e-06   (relative 3.7e-06)
MATCH - same result, half the weights per GPU.
```

The script passes when the relative error is below 1e-3, and 3.7e-06 clears that by a factor of about 270. The split is correct.

**Explanation.**
Any nonzero error here is just floating point. Both paths do the same maths but they add the numbers in a different order, and FP32 addition is not associative. The single-GPU reference does one big matmul over the full 16384-wide hidden dimension. The parallel version does two half-size matmuls and then adds the two partial results in the all-reduce. Different grouping, slightly different rounding, tiny difference. If the error were large it would mean the split itself was wrong, not that precision was lost.

**Why column-parallel needs no communication.** The first layer splits W1 by columns, so rank 0 owns columns 0 to 8191 and rank 1 owns 8192 to 16383. Both ranks already have the full input `x`. Each column of the output only depends on the full `x` and on that one column of W1, so each rank can compute its half of the hidden activations completely on its own. GELU is elementwise, so it also applies cleanly to a half. Nothing is missing, so nothing has to be sent.

**Why row-parallel needs an all-reduce.** The second layer splits W2 by rows, to line up with the hidden split that already exists. Rank 0 holds rows 0 to 8191 and multiplies them by its half of the hidden vector. That gives an output tensor of the correct final shape (8, 4096), but it is only a **partial sum**: it is missing the contribution of the other 8192 hidden units, which live on rank 1. The true output is the sum of both partial results. An all-reduce adds them and gives every rank the complete answer.

That pairing is the Megatron trick. Column-parallel then row-parallel means the whole MLP costs exactly **one** collective, at the very end, instead of one after each of the two layers.

**Is 3.7e-06 the right size for rounding noise?** Worth one line, because it shows the error is numerical and not a bug. FP32 machine epsilon is 5.96e-08, and the reduction being reordered runs over 8192 elements. Errors that add up like a random walk give sqrt(8192) x eps = 5.4e-06, and the absolute worst case is 8192 x eps = 4.9e-04. The measured 3.7e-06 sits inside that range and close to the random-walk estimate. That is ordinary rounding, exactly what you expect.

**The economics of TP, in numbers from this lab.** This is the part worth writing down, because it explains why anyone bothers with tensor parallelism:

```
W1 full   = (4096, 16384) FP32 = 268.4 MB      shard = 134.2 MB
W2 full   = (16384, 4096) FP32 = 268.4 MB      shard = 134.2 MB
per GPU   : 536.9 MB  ->  268.4 MB             saved 268.4 MB

the ONE all-reduce = y_partial (8, 4096) FP32 = 131 KB
```

Each GPU saves 268 MB of weights and pays 131 KB of communication, a ratio of about **2000 to 1**. That is the whole idea. Activations are small and weights are large, so you move the small thing and split the big thing.

**But this also shows the limit of TP, using the 6.7 GB/s we measured in Lab J.** 131 KB at 6.7 GB/s is only 20 microseconds of actual transfer, yet Lab J showed a fixed cost of about 80 microseconds per NCCL call. So a TP all-reduce is **latency-bound, not bandwidth-bound**. Making the link faster barely helps. And you pay that latency twice per transformer layer, in forward and again in backward, with nothing else to run while you wait, because the next operation needs the result. On a real model with 60 layers that is hundreds of small blocking calls per step. This is the concrete reason TP has to sit on NVLink inside a node, where the per-call latency is far lower, and it connects straight to Part 4 question 3 and to the `link_penalty = 18` cliff in the Part 5 Arena.
