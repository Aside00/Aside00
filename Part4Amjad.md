# Part 4
by: Amjad althobaiti

## Q1 :Why do 2 GPUs give ~1.7×, not 2× — and what makes that gap worse on the 5090 than on an H100 node?


## ANS:1

Because the second GPU is not free. DDP has to all-reduce the gradients every step so both model copies stay the same, and that time is communication, not compute. PyTorch hides most of it behind the backward pass, but the last gradient bucket and the sync barrier cannot be hidden. The gap is worse on the 5090 because consumer cards have no NVLink, so the all-reduce crawls over PCIe, while an H100 node has NVLink and NVSwitch at around 900 GB/s. Same amount of data, a link that is more than ten times slower, so much less of it fits inside the compute time.

## Q2 :DDP vs FSDP: what does each split, and how do you choose?

## ANS 2

DDP splits only the **data**. Every GPU keeps a full copy of the weights, gradients and optimizer states, and gradients are all-reduced each step. FSDP splits the **model state** too: parameters, gradients and optimizer states are each sharded across GPUs, gathered layer by layer when needed, and reduce-scattered in backward. Choose DDP when the model fits comfortably on one GPU, because it moves less data and is simpler. Choose FSDP when it does not fit, and accept that you pay extra communication to gather weights on demand.

## Q3:Why must tensor parallel live inside a node while data parallel can span the slowest links?

## ANS 3

TP does an all-reduce on activations inside every single layer, several times per step, and it sits directly in the critical path so nothing else can run while it waits. That volume only works on NVLink. DP is the opposite: one gradient all-reduce per step, and PyTorch overlaps it with the backward pass, so the latency is mostly hidden anyway. When a collective is rare and hideable you can afford to put it on the slowest link in the cluster.

## Q4 :What is the pipeline bubble, and how do you shrink it?

## ANS:4

The bubble is the idle time when pipeline stages have nothing to work on. At the start of a step the later stages wait for the first micro-batch to reach them, and at the end the earlier stages sit idle while the last one drains. You shrink it by using more micro-batches, so the fill and drain are a smaller share of the step, and by using a better schedule such as 1F1B, interleaved stages, or DualPipe. Using fewer pipeline stages also helps, if memory lets you.

## Q5:Why is 3D the name — what are the three coordinates of a GPU?

## ANS :5

Because every GPU has three coordinates, one per axis: its TP rank, its PP rank, and its DP rank. Together they form a 3D grid, and total GPUs = TP x PP x DP. Knowing a GPU's three coordinates tells you which slice of each layer it holds, which layers it holds, and which slice of the data it sees.

## Q6 : At 1,000 GPUs, why is checkpointing non-negotiable, and what does a recovery actually do?

## ANS 6

At 1,000 GPUs something breaks often. Each GPU, NIC, cable and power supply has its own failure rate, and multiplied across the cluster the mean time between failures drops to hours. One dead GPU kills the whole job, because all ranks are locked together in the collectives. Without checkpoints you would lose days of work. A recovery relaunches the job, has every rank load the last checkpoint (weights, optimizer states, RNG state and data loader position), rebuild the process group through rendezvous, and continue from that step instead of from zero.

## Q7: What does DCGM give you that nvidia-smi on one box does not?

## ANS 7

`nvidia-smi` gives you a snapshot of one machine, and only while you are looking at it. DCGM collects telemetry across every node in the cluster continuously: utilization, SM occupancy, memory, power, temperature, NVLink and PCIe errors, ECC counters and XID errors. It also runs active health checks and exports to Prometheus, so you can alert on a failing GPU and see history. On 1,000 GPUs you need to know *which* node is the straggler, and `nvidia-smi` cannot tell you that.

---
