# Replace the stalled Gemma diagnostic using current compatible GPU capacity

The reported queue has job 1200227 pending for a 32 GB V100 FP32 six-case test.
The node snapshot shows idle H200s on evc105, unreliable evc11 V100 allocations,
idle 16 GB V100s that cannot hold the 4B model's FP32 weights, and several mixed
H100/32 GB V100 nodes. State `MIXED` alone does not establish free GPU/CPU/RAM;
`PLANNED` nodes may be held for higher-priority allocations. For example evc21's
configured unallocated RAM is 190884 - 155744 = 35140 MB, less than a 48/64 GB request.

`../scripts/submit_available_gpu_smoke.py` queries live node metadata and:

1. Requires at least one unallocated GPU, two CPU slots and 32 GiB unallocated
   Slurm-configured host memory; ignores OS FreeMem as an allocation guarantee.
2. Excludes the observed problem nodes evc11/evc26, unavailable states and planned nodes.
3. Prefers evc105 H200 through `short`/`ucfit`, then healthy normal H100s, then
   normal 32 GB V100s for the two 4B experts. It does not use 16 GB V100s or
   select a V100 for Gemma 4 FP32.
4. Checks each candidate's admission with `sbatch --test-only` using account ytian.
   Inaccessible partitions are skipped; it does not try another person's account.
5. Compares start estimates returned by successful admission checks when available;
   unknown estimates fall back to the GPU-family preference. Submits one
   20-minute BF16 H100/H200 test, or a one-hour FP32 V100 test, with
   SMOKE=1 and separate precision-specific results. Existing environments are reused.
6. Cancels only the old job explicitly supplied by the caller, after Slurm accepts
   the replacement. Rejected submissions never cancel the old job.

This does not bypass Slurm priority/reservations or guarantee immediate allocation,
and cannot establish GPU driver health until the actual preflight runs. It submits
a six-case diagnostic only, not full expert generation or GRPO training. The BF16
route returns to the original successful H100 precision rather than the empty FP16
outputs observed on V100. FP32 V100 fit/performance remain experimental.

## Run on Newton's login node

```bash
cd "$HOME/Fairvision-gemma-RETFound"
git -c pack.threads=1 -c index.threads=1 pull --ff-only
"$HOME/.venvs/fairvision-cpu-py311/bin/python" equi-agent/scripts/submit_available_gpu_smoke.py --replace-job 1200227 --model gemma
```

The helper uses only Python's standard library and Slurm commands; no installation
or GPU model loading occurs in the login shell. It prints admission errors, the new
job number, chosen node/partition, precision, and the exact log path. Follow that log
and check `sacct` for the actual printed ID. Do not copy a literal placeholder.

If no candidate is accepted, it prints the reason and retains the old queue entry.
That means the visible hardware is not currently schedulable through these allowed
requests; permissions/priority need resolving rather than repeated cancellation.
If a new job is accepted but cancellation/status inspection fails, both IDs are
reported for manual inspection. No unrelated jobs are cancelled.

The helper supports `gemma`, `medgemma` and `gemma4-it` when an explicit old job
is supplied. Full inference should follow only after valid predictions, zero errors,
and exit `0:0`. Keep per-expert checkpoint/precision/runtime consistent across all
GRPO splits, as enforced by the current router bundle format.

Local tests use a mocked Slurm CLI to validate capacity filtering, partition fallback,
precision selection and cancellation order. No real cluster request was submitted
by the assistant during those tests.
