# Tutorial 2 Evidence Log

Use this log while running ATP. Keep entries short and tied to screenshot filenames.

## Run metadata

- Date:
- Machine (instance type / CPU):
- Compiler + version:
- Build flags:
- ATP version:
- Command args used (`N`, `iters`):

---

## Baseline (`triad_baseline`)

- Screenshot: `assets/t2_baseline_recipe_setup.png`
- Observation:

- Screenshot: `assets/t2_baseline_functions.png`
- Observation:

- Screenshot: `assets/t2_baseline_source.png`
- Observation:

- Console result:
  - Time (ms):
  - Bandwidth (GB/s):
  - Checksum:

---

## Restrict (`triad_restrict`)

- Screenshot: `assets/t2_restrict_recipe_setup.png`
- Observation:

- Screenshot: `assets/t2_restrict_source.png`
- Observation:

- Screenshot: `assets/t2_restrict_disassembly.png`
- Observation:

- Console result:
  - Time (ms):
  - Bandwidth (GB/s):
  - Checksum:

---

## Aligned (`triad_aligned`)

- Screenshot: `assets/t2_aligned_recipe_setup.png`
- Observation:

- Screenshot: `assets/t2_aligned_source.png`
- Observation:

- Screenshot: `assets/t2_aligned_memory_view.png`
- Observation:

- Console result:
  - Time (ms):
  - Bandwidth (GB/s):
  - Checksum:

---

## Final comparison (copy into instructions/report)

| Variant | Hot line instruction pattern | Key inspector observation | Time (ms) | Bandwidth (GB/s) |
|---|---|---|---:|---:|
| Baseline |  |  |  |  |
| Restrict |  |  |  |  |
| Aligned |  |  |  |  |

## One-paragraph summary draft

Using ATP Memory Access with Source Code Inspector, we mapped a source-level aliasing/alignment change to a measurable change in memory instruction behavior on the same hot line. The baseline showed [scalar pattern], adding `__restrict__` changed this to [vector pattern], and adding alignment assumptions preserved/improved this behavior with [alignment observation]. This correlated with runtime/bandwidth moving from [X] to [Y] to [Z].
