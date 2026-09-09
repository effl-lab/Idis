"""Prompts used in the paper (Appendix B.4).

  *_BASE     : task prompt (main experiments)
  *_STRATEGY : attribute-guiding prompt (Sec. 6.3)
  PROMPTS[dataset][prompt] ← run_perception.py --dataset ... --prompt {base,strategy}
"""

# ── Idis-perception ────────────────────────────────────────────────────────
PERCEPTION_BASE = (
    "[Question] Which category best describes the main object in the image? "
    "Choose exactly one from: Dog, Bird, Vehicle, Reptile, Carnivore, Insect, "
    "Instrument, Primate, Fish.\n"
    "Use a thinking process to analyze the problem step-by-step.\n"
    "At the end, provide your answer and clearly indicate it using <answer>X</answer> format."
)

# Appendix B.4 "Prompt strategy".
PERCEPTION_STRATEGY = (
    "[Question] Which category best describes the main object in the image? "
    "Choose exactly one from: Dog, Bird, Vehicle, Reptile, Carnivore, Insect, "
    "Instrument, Primate, Fish. \n"
    "First identify the main object. Base your reasoning only on that object's own "
    "visual attributes, and determine which of the nine categories best matches the "
    "object. Do not assume that the most visually dominant object is the main object. "
    "Do not infer the category from surrounding objects or from items that are merely "
    "associated with another category.\n"
    "At the end, provide your answer as <answer>X</answer>."
)

# ── Waterbirds ─────────────────────────────────────────────────────────────
WATERBIRDS_BASE = (
    "[Question] Is the bird in the image a waterbird or a landbird?\n"
    "Use a thinking process to analyze the problem step-by-step.\n"
    "At the end, provide your answer and clearly indicate it using <answer>X</answer> format."
)

WATERBIRDS_STRATEGY = (
    "[Question] Is the bird in the image a waterbird or a landbird? \n"
    "Think step by step based on the foreground bird's attributes. \n"
    "At the end, select your answer from the provided options and clearly indicate it "
    "using <answer>X</answer> format."
)


# ── lookup used by run_perception.py ───────────────────────────────────────
PROMPTS = {
    "Background_Challenge": {"base": PERCEPTION_BASE, "strategy": PERCEPTION_STRATEGY},
    "waterbirds":           {"base": WATERBIRDS_BASE,  "strategy": WATERBIRDS_STRATEGY},
}
