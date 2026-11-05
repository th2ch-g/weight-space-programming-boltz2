# Weight-space programming for Boltz-2 ligand geometry

This project tests direct edits to Boltz-2 weights using the reasoning-driven
method described in [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/).
The experiment uses the ct209 dataset from the adjacent `bench-rgi` workspace.
It does not train the model, use inference guidance, or repair output coordinates.

The [edited checkpoints are distributed on Hugging Face](https://huggingface.co/wasarou/weight-space-programming-boltz2),
with a model card, all thirty edit specifications, evaluation summaries, and
SHA-256 checksums. They are experimental checkpoints; none is a validated repair.
The upstream Boltz-2 license is retained in
[third-party notices](THIRD_PARTY_NOTICES.md).
The source code in this repository is released under the [MIT license](LICENSE).

This source repository excludes the blog manuscript, private session transcript,
checkpoints, prepared inputs, and generated predictions. The public ligand dataset
is [PLINDER cis/trans 209](https://huggingface.co/datasets/wasarou/plinder-cistrans-209).
Reproducing the archived campaign additionally requires its prepared benchmark
inputs and a compatible `bench-rgi` checkout for the legacy `common.chiral`
metrics. That adjacent workspace is not vendored here; the evaluator accepts its
location through `--bench`. Modern CIP and full E/Z evaluation are implemented
locally in `scripts/evaluate.py`.

Prepare the environment with `uv sync --frozen` and run the CPU test suite with
`uv run --frozen pytest tests`. The Slurm workflows below describe the recorded
experiments and keep generated artifacts outside version control.

## Completed follow-up evaluations

The initial campaign described below is historical. Its 104-system test split
has now been used for the frozen-v2 and v11 evaluations. The split membership remains
55 development, 50 validation, and 104 test systems; test results were not used
to edit or select the thirty retained candidates.

Three frozen studies completed 3,153 new predictions without execution or
evaluation failures, using at most four simultaneous GPU nodes:

- [Frozen v2 evaluation](https://huggingface.co/wasarou/weight-space-programming-boltz2/blob/main/evaluation/frozen_v2_evaluation.md): reuse the
  original 55 development pairs, evaluate 50 validation pairs at seed 1, and
  evaluate all 104 test pairs at seeds 1, 2, and 3. On validation, R/S agreement
  decreased from 27/50 to 25/50. Across the three test seeds, joint agreement
  was unchanged at 103/312 predictions, and every primary paired-effect
  confidence interval included zero. Repeated seeds were averaged within each
  system before resampling sequence/scaffold groups.
- [Matched comparison of all thirty candidates](https://huggingface.co/wasarou/weight-space-programming-boltz2/blob/main/evaluation/all_candidates_development.md):
  rerun the baseline and all thirty checkpoints on the same 55 development
  systems at seed 1. Each system's 31 runs share a GPU allocation, input,
  software, and inference settings. v11 had higher R/S and joint agreement
  than v2, equal E/Z agreement, and lower bond and angle medians. However,
  v11's bond median was worse than the baseline. No candidate met the
  predeclared simultaneous-improvement criteria. This is a development
  comparison and does not establish a validated replacement.
- [Frozen v11 evaluation](https://huggingface.co/wasarou/weight-space-programming-boltz2/blob/main/evaluation/frozen_v11_evaluation.md):
  reuse the matched 55 development pairs and run another 724 predictions for
  validation and test, with the same split and seeds as the v2 follow-up.
  Validation R/S agreement decreased from 27/50 to 25/50. Across the three
  test seeds, R/S agreement changed from 125/312 to 129/312, full E/Z from
  233/312 to 230/312, and joint agreement remained 103/312. Every primary
  pooled-effect confidence interval included zero. All 362 fresh baseline
  outputs were byte-identical to the corresponding v2-study baselines.
  The test split had already been used for v2; this is a follow-up comparison,
  not a new untouched-test confirmation. The original repair gate remains unmet.

The old v2 development table is reproducible from its saved predictions, but
its small bond-median improvement did not persist in the fresh comparison.
The [cross-run check](https://huggingface.co/wasarou/weight-space-programming-boltz2/blob/main/evaluation/reproducibility_check.json)
found byte-identical predictions for both checkpoints in all 31 systems that
retained the same GPU family. Output differences occurred in the other 24
systems, whose GPU family changed. This identifies a reproducibility limitation;
it does not isolate the GPU from every other machine-environment difference.

Each study retains its frozen protocol, source snapshot, checkpoint hashes,
per-system scores, paired-condition audits, and scheduler audit. The original
requirement of zero R/S and E/Z errors across all 209 systems remains unmet.
The blog manuscript is maintained separately and excluded from this repository.

## Initial campaign (historical)

The initial four-candidate experiment **did not establish a successful repair**. The selected
candidate changed joint stereochemistry agreement from 20/55 to 22/55 in
development and from 22/50 to 23/50 in validation. However, validation chirality
agreement fell from 27/50 to 25/50, and the median paired protein CA RMSD was
0.703 A, exceeding the frozen 0.5 A acceptance limit. The 104-system test split
remains unused. Those candidate checkpoints are retained in the project root;
none was promoted to a validated replacement.

Work continues under `goal_protocol.json`. The user requires zero R/S and E/Z
errors across the full 209-system dataset, together with lower bond-length and
bond-angle error than the paired base checkpoint. The zero-error requirement was
explicitly retained in the latest clarification. Protein preservation remains a
secondary check. The previous validation
results are known and cannot serve as an untouched evaluation of later edits.
New candidates remain experimental until the full acceptance criteria pass.

The first 26 candidates all failed the zero-error requirement. In the
eight-system pilot, v22, v23, and v24 preserved all R/S centers in 4, 4, and 3
systems, respectively; each preserved all E/Z bonds in 6 systems. Their median
bond-length errors worsened. The actual encoded reference conformers were
correct for both R/S and E/Z in all eight systems, so input stereochemistry
errors do not explain those failures.

The broader v15 development comparison completed all 55 paired systems. R/S
complete agreement stayed at 21/55, full E/Z agreement changed from 36/55 to
37/55, and joint agreement stayed at 20/55. Correct R/S centers decreased from
169/267 to 163/267. Median bond-length error increased from 0.038337 to
0.038517 A, while median bond-angle error decreased from 2.9074 to 2.7716
degrees. The reported paired confidence intervals include zero. Thus the
eight-system partial gains did not establish a broad repair. Protein CA
displacement exceeded 0.5 A in 16 systems, with a maximum of 6.396 A; every
outlier is listed in the generated report. This comparison remains development
data and includes the original eight pilot systems.

The frozen settings and evaluation rules are in `protocol.json`. The data split
groups related protein sequences and ligand scaffolds before any prediction is
evaluated. Development, validation, and test contain 55, 50, and 104 systems.
Candidate checkpoint files are not validated repairs until independent evaluation
supports that conclusion.

One ct209 molecule exceeds upstream Boltz's four-character atom-name limit.
`scripts/prepare_overflow.py` reproduces the benchmark's per-element naming
fallback only for that molecule, then stores the same molecular graph and native
reference conformer in a local CCD. Both comparison arms use that immutable
input. The other 208 molecules retain their original naming. Initial parser
failures remain recorded separately from the successful retries.

Create the dedicated environment with `UV_CACHE_DIR=.cache/uv uv sync --frozen`.
All Python commands use `uv`. The Slurm wrapper defaults to this project's
`.venv/bin/python`. GPU prediction and substantial data analysis run on compute
nodes. Supply the cluster's candidate partitions through `GPU_PARTITIONS`.

Set `BENCH_CONFORMER_ROOT` to the existing ct209 campaign directory and
`BOLTZ_CACHE` to the original Boltz cache. Set `CPU_PARTITIONS` for CPU jobs.
The original cache is only read.

```bash
mkdir -p logs
sbatch --wait --partition="$CPU_PARTITIONS" --cpus-per-task=4 --mem=24G \
  --output='logs/prepare-%j.log' scripts/slurm.sh scripts/prepare.py \
  --dataset "$BENCH_CONFORMER_ROOT" --checkpoint "$BOLTZ_CACHE/boltz2_conf.ckpt"
UV_CACHE_DIR=.cache/uv uv run --frozen python scripts/link_cache.py "$BOLTZ_CACHE"
sbatch --wait --partition="$CPU_PARTITIONS" --cpus-per-task=4 --mem=8G \
  --output='logs/input-fix-%j.log' scripts/slurm.sh scripts/prepare_overflow.py
sbatch --wait --partition="$CPU_PARTITIONS" --cpus-per-task=4 --mem=24G \
  --output='logs/edit-%j.log' scripts/slurm.sh scripts/edit_weights.py \
  edits/input_geometry_v1.json --output boltz2_wsp_input_geometry_v1.ckpt
UV_CACHE_DIR=.cache/uv uv run --frozen python scripts/make_plan.py \
  --split development --pilot \
  --checkpoints boltz2_base.ckpt boltz2_wsp_input_geometry_v1.ckpt \
  --output artifacts/pilot.json --run-root artifacts/pilot
UV_CACHE_DIR=.cache/uv uv run --frozen python scripts/submit.py artifacts/pilot.json
```

`scripts/submit.py` requests at most four one-node GPU tasks and waits behind any
existing project GPU jobs through Slurm dependencies. The worker keeps each
complex's comparison arms on the same node. When temporary space is available,
it uses verified, read-only checkpoint copies for inference and removes them
when the worker exits; canonical edited checkpoints stay in the project root.

After prediction completes, submit `scripts/evaluate.py artifacts/pilot.json`
and `scripts/audit_pairs.py artifacts/pilot.json` through the CPU wrapper. Use
`scripts/summarize.py artifacts/pilot.scores.json --stage development --output
artifacts/pilot_summary` to produce the paired report. For larger systems,
choose GPU partitions with sufficient memory and split plans by protein length.

Only one GPU array may be active for this project at a time. Each array has at
most four one-node tasks. CPU preparation and evaluation use the same wrapper
without a GPU request. Preserve failed run directories and create a new output
directory for retries.

`scripts/prepare.py` imports and hashes inputs and copies the original structure
checkpoint into `boltz2_base.ckpt`. The dataset index contains historical absolute
paths; imports resolve the current dataset location and verify the stored hashes.
`scripts/edit_weights.py` writes a new checkpoint in the project root and checks
every untouched tensor. Editing specifications in `edits/` record the hypothesis
and exact operations. Original external checkpoints are never modified.

Generated data, checkpoints, environments, caches, and logs are ignored. Keep
the scripts, editing specifications, protocol, and `uv.lock` to reproduce them.

The primary chirality evaluator uses modern CIP assignments and graph
isomorphisms. The legacy benchmark chirality result is reported separately.
For continued work, `full_ez_all` evaluates all defined E/Z bonds readable from
heavy-atom coordinates, including ring bonds. The original `ez_all` metric follows
the RGI benchmark and excludes ring bonds. Both are retained, and bonds whose
E/Z cannot be determined without hydrogen information are counted explicitly.
`full_stereo_all` requires a single graph mapping to preserve both R/S and E/Z.
Paired ligand displacement is measured after fitting the protein; it is not a
measure of experimental pose accuracy. The editing split does not establish
that the original Boltz-2 training data excluded these complexes.

Boltz 2.2.1 leaves `contact_guidance_update=True` when `use_potentials=False`.
The runner explicitly disables contact guidance, physical guidance, and FK
steering when loading either checkpoint. This records the actual settings
behind the no-guidance comparison.

The reference-parity diagnostics compare proper rotations and reflected reference
conformers to measure sensitivity to handedness. For the end-to-end diagnostic,
`scripts/predict.py --reference-reflection --diagnostic` reflects only ligand
reference coordinates before model encoding. The saved run explicitly records
this intervention. Diagnostic plans and reflected runs are excluded from
benchmark scoring; `scripts/analyze_prediction_parity.py` analyzes their response.
These counterfactual predictions cannot satisfy the repair acceptance criteria.

The later reference-path diagnostic separates the input/trunk and denoiser
paths with `--reflection-path input` or `--reflection-path denoiser`. It restores
the original coordinates immediately after the selected encoder and records
the coordinates seen by both encoders. In three development systems, all 30
output R/S labels stayed unchanged when either path or both paths were
reflected. Denoiser-only ligand displacement was below 0.005 A. These results
do not support cancellation between the two paths, but they do not prove that
every possible weight edit must fail. A separate audit confirmed correct
encoded R/S and E/Z in all 55 development references and identical ligand
reference coordinates in their paired base/v15 runs.

Additional CPU diagnostics test the existing nonlinear computations.
`scripts/prove_cubic_parity.py` constructs a signed-volume calculation using
twelve neurons in the unchanged Boltz transition class. Its synthetic tests
establish a representable calculation, not a working edit to the pretrained
model. Preserving the pretrained representation while supplying neighbor vectors
and connecting the scalar to the learned decoder remain unresolved. The separate cubic-limit probe did not isolate a
strong parity signal and was not promoted to a candidate checkpoint.

The v25/v26 controlled experiment programs a reference-distance prior into two
existing pair channels and four existing pair-MLP units. The v25 checkpoint
adds that prior to the first input attention layer; v26 reserves the same
capacity with zero added prior. `scripts/program_neighbor_bias.py` specifies
the analytic weight assignments. In the eight-system comparison, v25 had 3/8
R/S and 6/8 E/Z complete agreements, whereas the base and v26 each had 4/8 and
6/8. The local attention prior did not repair stereochemistry. A separate CPU
diagnostic tests stronger neighborhood masking; it has not produced a
checkpoint or an accepted prediction result. In that diagnostic, near-neighbor
attention reached approximately 100 percent at the two inspected centers, but
the parity-to-rotation ratios worsened in all three probed ligands. The strong
mask was not promoted to a checkpoint.

The existing affinity checkpoint does not provide a different encoder: all 106
inspected input, denoiser-conditioning, and bond-type tensors match the structure
checkpoint exactly. A separate polar-factor alignment between the input and
denoiser chemical embeddings leaves 98--99 percent residual on chemical columns
excluded from the alignment. Neither diagnostic supports transferring those
coordinate weights into the denoiser.

The later constructive diagnostics assign weights analytically in unchanged
Boltz classes. `scripts/program_reference_parity.py` routes local coordinate
moments through the atom encoder and computes their signed volume. It passes
rotation and reflection sign checks at 29 of 30 inspected centers. At the remaining
center, one directly bonded atom lies outside the attention window. Moving the
calculation to global Pairformer attention makes all 30 centers pass in float32,
but the three-head version remains unreliable in bfloat16. The numerical audit
identifies both collapsed atom-name codes and near-zero centroid vectors at
four-neighbor centers.

`scripts/program_parity_vector.py` keeps the four character codes separate,
uses eight attention heads biased by actual bond adjacency, and programs two
existing transitions to compute 56 signed-volume components. The fixed program
passes the diagnostic at all 267 defined centers in the 55 development systems,
in both CUDA float32 and bfloat16. Each precision includes 6,408 proper rotations,
6,408 reflected rotations, and 534 translations. The report is
`artifacts/programmed_parity_vector_development_summary.json`. The component
with largest magnitude in each float32 reference is selected outside the model
for this diagnostic; that selection is not an implemented inference component.
The test split remains unused.

These programs replace the tested modules and do not preserve their pretrained
representations. Their signs are not CIP labels, and they are not connected to
the coordinate decoder. They therefore establish a computable parity feature,
not a repaired checkpoint or improvements in R/S, E/Z, bond length, or bond angle.
Synthetic tests independently compare the programmed outputs with determinants
for three- and four-neighbor centers and continuous rotations.

`scripts/validate_parity_detector.py` compares all 56 reference and predicted
parity components using a fixed zero threshold on their dot product. On the
55 saved development systems, it detects all 98 baseline errors and all 104 v15
errors among 267 defined centers, with no false positives. Float32 and CUDA
bfloat16 agree across three proper rotations. This diagnostic uses named-center
CIP labels and modifies no saved coordinates or checkpoints. The comparison is
error detection, not chemistry repair.

The v27 control reserves eighteen atom channels and eighteen denoiser token
channels by zeroing their learned sources, readers, and residual updates. The
ranking uses only original checkpoint readout norms. It does not use structures,
labels, or prediction scores. `scripts/program_reserved_channels.py` implements
this control, and a synthetic test verifies that the reserved states stay zero
through the unchanged Boltz forward pass. LayerNorm statistics still couple the
reserved and retained states, so this edit is not function preserving. Paired
prediction evaluation is required before adding any parity calculation.

The paired v27 pilot completed all eight systems, and all pair audits passed.
R/S complete agreement fell from 4/8 to 2/8, E/Z from 6/8 to 4/8, and joint
agreement from 4/8 to 1/8. Median bond-length error increased from 0.03895 to
0.37269 A, while median bond-angle error increased from 2.8983 to 18.7613
degrees. Every system worsened in both geometry metrics, and every protein CA
displacement exceeded 0.5 A. This reservation scheme was rejected as a basis
for adding the parity program. The v28 and v29 controls isolate the same
atom-only and token-only reservations to identify the source of the damage.
Both isolated controls also worsened bond-length and bond-angle error in every
pilot system. Atom-only v28 had R/S 5/8, E/Z 6/8, median bond error 0.09501 A,
and median angle error 5.6015 degrees. Token-only v29 had R/S 3/8, E/Z 6/8,
median bond error 0.18108 A, and median angle error 8.4659 degrees. Their
paired baseline had R/S 4/8, E/Z 6/8, bond error 0.03895 A, and angle error
2.8983 degrees. All sixteen pair audits passed. These results reject both
reservations as a preserved starting point; no signal-carrying extension was
promoted.

A separate float64 diagnostic, `scripts/probe_cross_moment_parity.py`, compares
reference and predicted homogeneous cross moments. The determinant identity
reduces the comparison to one 4-by-4 determinant, without atom-name codes or
56 separate parity components. Fixed center weights of 1 and 1/256 both match
named-center CIP correctness at all 267 centers in the saved baseline and v15
development predictions. They also match the v28 and v29 pilot labels, but
each misses two v27 centers whose predicted CIP labels are undefined. A signed
volume alone therefore does not certify assignable stereochemistry. Independent
tests verify the Cauchy-Binet identity and construct a four-neighbor counterexample
to a universal sign guarantee. No mixed-precision Boltz implementation or repair
checkpoint has been built from this smaller formula.

Read-only activation profiles (`scripts/trace_reservation.py`) show that the
eighteen selected token channels account for roughly 25 percent of centered
variance at the first denoiser token layer in three inspected ligands. Zeroing
them reduces the LayerNorm standard deviation to about 0.864--0.865 of the
original value. Low final readout norms therefore did not identify negligible
early states. The earlier reservation also deleted conditional-bias contributions
after AdaLN; that effect is distinct from residual-state removal.

The v30 control retains those conditional biases while suppressing the selected
residual inputs through the existing AdaLN scale gates. It uses the same channels
as v29. All sixteen paired audits passed, but v30 also failed: R/S complete
agreement was 2/8, E/Z 3/8, and joint agreement 0/8. Median bond-length and
bond-angle errors were 0.35769 A and 14.3840 degrees, compared with 0.03895 A
and 2.8983 degrees for the paired baseline. Every system worsened in both
geometry metrics. Preserving the conditional biases alone did not rescue the
reservation, and no signal-carrying extension was created.

A checkpoint-only audit examines whether paired channels have matching source,
writer, gate, and conditional-bias rows. The smallest normalized mean squared
row distances are 0.8200 for atom channels and 0.4379 for token channels; exact
duplicates would have distance zero. These similarities do not establish a
function-preserving merge. No paired-channel checkpoint was created. Static
source bounds also do not justify replacing the inspected states with constants.

The first broad parity diagnostic stopped after 27 systems because its feature
loader used the global molecule cache for the overflow-named ligand. The loader
now uses the same local CCD cache as the saved prediction and verifies the stored
molecule hash. The original failure is retained, and the remaining 28 systems
were retried without changing the programmed weights. Production predictions
already used the local CCD. `scripts/monitor_gpu_jobs.py` provides bounded live
monitoring; it expands recorded array submissions into all expected task IDs
before checking completion. The initial array monitor incorrectly expected the
parent ID in accounting output. Its observations were retained, and the fixed
monitor was resumed without restarting predictions. The accounting audit
independently verifies the full recorded allocation history.

After evaluation, submit `scripts/verify_artifacts.py --original-checkpoint
"$BOLTZ_CACHE/boltz2_conf.ckpt"` through the CPU wrapper. It verifies the original
and local checkpoint hashes, the recorded comparison audits, and that no test
systems occur in the saved plans. The result is `artifacts/final_integrity.json`;
passing this integrity check does not mean that ligand repair succeeded.

## Publishing checkpoint artifacts

`scripts/prepare_weight_release.py` checks all thirty checkpoint hashes and
creates an ignored staging directory using hard links. It includes the edit
specifications, upstream license, model card, checksums, and completed evaluation
summaries. Run this file verification in a CPU allocation on a shared cluster.
The publication helper uses a separate Hugging Face environment so the frozen
inference environment stays unchanged.

```bash
uv run --frozen python scripts/prepare_weight_release.py \
  --output artifacts/hf_release \
  --github-url "$GITHUB_REPOSITORY_URL" \
  --model-id "$HF_MODEL_ID" \
  --upstream-revision "$UPSTREAM_COMMIT"
uv run --no-project --with huggingface_hub==1.27.0 python \
  scripts/upload_weight_release.py artifacts/hf_release \
  --repo-id "$HF_MODEL_ID" --source-commit "$(git rev-parse HEAD)"
```

Use a new staging directory for a new release. The uploader retains resumable
upload state and verifies the remote size and SHA-256 of every checkpoint.
Blog and session files are excluded from the source repository.
