# Reconstruction learning curves

**Status:** G1 and its one retest did not pass, so [the diffusion track has stopped](g1-four-pairs.md). Real training needs passed G1 and G2 results, so the code refuses it. The tooling remains, tested on fixtures only.

`nde reconstruction-experiment register CONFIG --out EXPERIMENT` freezes a
versioned `experiment.json` before training. It binds local base weights,
approved training datasets, a separately authorized fixed held-out evaluation
dataset, seeds, ordered 25/50/100% subsets, objective pairs, optimizer settings,
and token/attention/time caps. Cardinality is checked before creating job files.
Each training dataset retains its registered whole-story/window/scene granularity
and masking schedules; evaluation tasks remain identical across every matrix entry.

Subsets are prefixes of one deterministic seeded **author** order, using ceil
rounding. Every work, edition, family and overlapping window from that author
stays together. A derived subset loader revalidates the complete approved parent
before reproducing selection; it never opens evaluation annotations for fitting.
Small fixtures can have identical memberships at different percentages; reports
retain the requested percentage and exact actual groups rather than inventing
additional independent stories.

`preflight EXPERIMENT` revalidates all frozen inputs without training.
`run EXPERIMENT` and `resume EXPERIMENT` execute one shared CPU engine serially.
`--max-jobs-this-call N` bounds a call; `--steps-per-call N` leaves resumable
partial checkpoints. An OS advisory lock is automatically released if a process
exits, so a leftover lock file cannot permanently block resume.
`report EXPERIMENT` reads the resulting report.

Each job reserves its full registered training **and inference** token caps before
launch. Completed jobs release unused allowance to their measured work. Failed
or unknown work keeps the entire reservation; checkpoint progress and failures
remain visible and cannot disappear from aggregate limits. Completed entries
are reused, and input changes reject resume. Optimizer steps also save a durable
work reservation before execution: work interrupted before checkpoint publication
is conservatively charged again on resume rather than silently resetting compute.
The reported failed-attempt reservation is separate from completed step counts.
Elapsed time includes saved progress and each bounded call; resource limits are
checked between finite operations rather than preempting a tensor kernel.

Every pair freezes the exact future prediction/score paths, actual derived dataset
and task IDs, seed, granularity and budget policy. Reports include every expected
job, partial/resource/failure statuses, checkpoint and prediction identities,
actual training/inference work and paired score paths. Invalid or resource-stopped
predictions remain in scoring denominators. Training/checkpoint artifacts remain
immutable; progress pointers and reports are the mutable views.

Pass human-only and augmented datasets as distinct `train_datasets` to preserve the
human control. Both objectives receive the same registered mixture within a pair;
source groups, generator costs and human-control linkage remain in reports.
Recipe/model selection is validation-only; fixed test results never select the
configuration later described as registered. Software fixtures remain test-only,
and no sweep result automatically passes G3 or launches Phase 7.


## Time allowance and interrupted work

Registration divides `max_total_seconds` into frozen per-job training and inference
portions. Their sum over the matrix fits the total allowance; a requested phase cap
can reduce its portion further. Candidate and comparator receive identical portions.
Each started job durably reserves both portions before executing, and that seconds
charge remains on completed, failed, partial and interrupted entries.

Training checks its saved phase timer before every optimizer step, includes setup,
and records a resource stop when a step crosses the cap. Inference checks a deadline
before each forward pass, records elapsed seconds and preserves every stopped task.
Checks occur between indivisible operations, so one kernel or file-validation
operation may overrun the deadline; that overrun is reported and cannot pass G3.
No timer can preempt a tensor kernel already executing.

Phase timer journals survive incomplete result publication. Unknown interrupted
inference consumes its remaining time portion conservatively, retaining known token
work and producing explicit resource stops on resume. Training combines its timer
with the existing durable optimizer-attempt journal. G3 checks each pair's allocated
inference token/time caps rather than the original unallocated request. Older saved
score fixtures without a time field keep their registered token-only interpretation.
