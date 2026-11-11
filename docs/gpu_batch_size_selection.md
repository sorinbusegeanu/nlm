# GPU training batch-size selection

Neural training paths with `auto_batch_size: true` use one shared CUDA probe for
UltraBERT and LoRA/QLoRA. The probe keeps the existing exponential search and
binary search, so it can select any positive integer microbatch up to the
configured ceiling. It builds dynamically padded batches from the longest
tokenized examples, then runs forward, backward, and optimizer steps using the
configured precision, optimizer, and checkpointing mode.

`batch_probe_safety_fraction` is the fraction of currently free VRAM the probe
may consume. Values from `0.90` through `0.95` retain approximately 10% through
5% of currently free VRAM, respectively. The old implicit 20% reserve has been
removed. An explicit byte reserve can still be passed by low-level callers.
The probe records peak allocated and reserved VRAM, estimated device occupancy,
padded and non-padding token counts, elapsed GPU-step time, examples/second,
and tokens/second. It validates cached safe-batch results, then benchmarks up
to four nearby physical batch sizes and selects the fastest safe candidate.

The code-level default gradient accumulation is 1; existing YAML accumulation
values remain explicit recipe overrides. A config may set
`target_effective_batch_size` to retain a particular
optimizer batch size as physical microbatch changes. The selected microbatch is
then the largest divisor of that target that fits; accumulation is derived from
the quotient. Without that target, configured accumulation is left unchanged.
Optimizer steps per epoch, warmup, and schedulers use the selected physical
batch multiplied by the resulting accumulation count. CUDA OOM recovery reduces
the microbatch and selects an exact divisor to preserve the prior effective
batch when possible.

Manifests retain the requested and selected physical batches, accumulation,
effective optimizer batch, probe measurements, and allocated/reserved GPU peaks.
The Current training dashboard distinguishes physical and effective batches
and displays the selected probe's VRAM and throughput measurements.
