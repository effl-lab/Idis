# Data

Images are **not** stored in this repository. This directory holds only small
metadata files that the build scripts consume.

- `idis_math/shape_labels/` — human-confirmed coarse concept labels for
  MathVerse testmini diagrams (Plane Geometry / Solid Geometry / Functions).
  Consumed by `idis_math/build/gen_visual_distractors.py` to pick aligned /
  conflicting distractor shapes.

External sources required to rebuild Idis-math:

| Source | Used for | Where |
|---|---|---|
| MathVerse testmini | base problems & diagrams | https://huggingface.co/datasets/AI4Math/MathVerse |
| LogicVista (table subset) | irrelevant visual distractors | https://github.com/Yijia-Xiao/LogicVista |

Pre-built Idis images: **TODO** (hosting link to be added).
