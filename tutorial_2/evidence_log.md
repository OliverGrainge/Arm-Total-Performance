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

## Final comparison (copy into instructions/report)

| Variant | Hot line instruction pattern | Key inspector observation | Time (ms) | Bandwidth (GB/s) |
|---|---|---|---:|---:|
| Baseline |  |  |  |  |
| Restrict |  |  |  |  |

## One-paragraph summary draft

Using ATP Memory Access with Source Code Inspector, we mapped a source-level aliasing change to a measurable change in memory instruction behavior on the same hot line. The baseline showed [scalar pattern], and adding `__restrict__` changed this to [vector pattern]. This correlated with runtime/bandwidth moving from [X] to [Y].
