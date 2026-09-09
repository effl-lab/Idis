#!/usr/bin/env python3
"""Visual distractor insertion for Idis-perception with Gemini 2.5 Flash Image (paper Sec. 3.2 / App. A.4).

  input     : one ImageNet-9 original class directory of .JPEG images
  prompt    : "Using the provided image of a {class}, please add {objects} to the scene. ..."
  objects   : objects.py - aligned / conflicting / irrelevant, first n of 4 (n = 1..4)
  output    : <out-root>/<class dir>/<n>/<semantic>/<stem>.png  (1024x1024)
  conflict  : the donor class is sampled per image, so one cell mixes many donors
  failures  : empty or errored responses are appended to <out-root>/<failed>, replayed with --retry-from
  auth      : GOOGLE_CLOUD_PROJECT + GOOGLE_APPLICATION_CREDENTIALS (Vertex AI), or GOOGLE_API_KEY
  usage     : python gemini_edit.py --image-dir .../original/00_dog --semantic aligned --n 4
"""

import argparse
import os
import random
from io import BytesIO

import jsonlines
from google import genai
from PIL import Image

from objects import IRRELEVANT, OBJECTS, SEMANTICS, class_from_dir, short_name

MODEL = "gemini-2.5-flash-image"
IMAGE_EXTS = (".jpeg", ".jpg", ".png")


def build_client(args):
    """Vertex AI when a project is given, otherwise the Gemini Developer API."""
    if args.vertex_project:
        return genai.Client(vertexai=True, project=args.vertex_project, location=args.vertex_location)
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit(
            "No credentials: set --vertex-project (or GOOGLE_CLOUD_PROJECT) for Vertex AI, "
            "or GOOGLE_API_KEY for the Gemini Developer API."
        )
    return genai.Client(api_key=api_key)


def pick_objects(class_name, n, semantic, rng):
    """The n distractor objects for one image; `conflicting` draws a donor class per call."""
    if semantic == "aligned":
        return OBJECTS[class_name][:n]
    if semantic == "irrelevant":
        return IRRELEVANT[:n]
    if semantic == "conflicting":
        donors = [k for k in OBJECTS if k != class_name]
        return OBJECTS[rng.choice(donors)][:n] if donors else OBJECTS[class_name][:n]
    raise ValueError(f"Unknown semantic: {semantic}")


def build_prompt(class_name, objects):
    objects_text = ", ".join(f"one {obj}" for obj in objects)
    return (
        f"Using the provided image of a {class_name}, please add {objects_text} to the scene. "
        "Ensure the changes are seamlessly integrated into the natural setting. "
        f"Do not modify or obscure the {class_name}."
    )


def extract_image(response):
    """First inline image in the response, or None if the edit came back empty."""
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return None
    content = getattr(candidates[0], "content", None)
    for part in getattr(content, "parts", None) or []:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            return inline.data
    return None


def edit_one(client, img_path, save_path, class_name, n, semantic, rng, tries):
    """Generate one edited image. Returns True once written, False if every try came back empty."""
    prompt = build_prompt(class_name, pick_objects(class_name, n, semantic, rng))
    source = Image.open(img_path)
    for attempt in range(1, tries + 1):
        data = extract_image(client.models.generate_content(model=MODEL, contents=[source, prompt]))
        if data:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            Image.open(BytesIO(data)).save(save_path)
            return True
        print(f"Empty response ({attempt}/{tries}): {img_path}")
    return False


def run(client, args, images, save_root, writer):
    """Edit every image in `images`, logging the ones that never produced output."""
    rng = random.Random(args.seed)
    done = failed = skipped = 0

    for img_path in images:
        save_path = os.path.join(save_root, f"{os.path.splitext(os.path.basename(img_path))[0]}.png")
        if args.skip_existing and os.path.exists(save_path):
            skipped += 1
            continue
        try:
            if edit_one(client, img_path, save_path, args.class_name, args.n, args.semantic, rng, args.tries):
                done += 1
                print(f"Saved: {save_path}")
                continue
        except Exception as e:  # transport, quota and safety-filter errors all land here
            print(f"Error processing {img_path}: {e}")
        writer.write({"image": os.path.abspath(img_path)})
        failed += 1

    print(f"{args.class_name} n={args.n} {args.semantic}: {done} saved, {failed} failed, {skipped} skipped")


def scan_images(image_dir):
    return [
        os.path.join(image_dir, f)
        for f in sorted(os.listdir(image_dir))
        if f.lower().endswith(IMAGE_EXTS)
    ]


def read_failed(path):
    """Source images from a failure log; missing files are reported and dropped."""
    images = []
    with jsonlines.open(path) as reader:
        for item in reader:
            img_path = item.get("image")
            if img_path and os.path.exists(img_path):
                images.append(img_path)
            else:
                print(f"Missing source, cannot retry: {img_path}")
    return sorted(set(images))


def main():
    ap = argparse.ArgumentParser(description="Insert visual distractors into ImageNet-9 images with Gemini")
    ap.add_argument("--image-dir", required=True,
                    help="one ImageNet-9 original class directory, e.g. .../original/00_dog")
    ap.add_argument("--out-root", required=True, help="dataset root; outputs go to <out-root>/<class dir>/<n>/<semantic>/")
    ap.add_argument("--semantic", choices=SEMANTICS, required=True)
    ap.add_argument("--n", type=int, choices=[1, 2, 3, 4], required=True, help="number of distractor objects")
    ap.add_argument("--class-name", default=None,
                    help="target class in the prompt (default: from --image-dir, `02_wheeled vehicle` -> `wheeled vehicle`)")
    ap.add_argument("--tries", type=int, default=1, help="generation calls per image before logging a failure")
    ap.add_argument("--seed", type=int, default=42, help="donor-class sampling for --semantic conflicting")
    ap.add_argument("--skip-existing", action="store_true", help="leave already generated outputs untouched")
    ap.add_argument("--failed", default=None,
                    help="failure log under --out-root (default: failed-<class>-<n>-<semantic>.jsonl)")
    ap.add_argument("--retry-from", default=None,
                    help="regenerate the source images listed in this failure log instead of scanning --image-dir")
    ap.add_argument("--vertex-project", default=os.environ.get("GOOGLE_CLOUD_PROJECT"),
                    help="GCP project for Vertex AI (env: GOOGLE_CLOUD_PROJECT)")
    ap.add_argument("--vertex-location", default=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
                    help="Vertex AI location (env: GOOGLE_CLOUD_LOCATION)")
    args = ap.parse_args()

    class_dir = os.path.basename(args.image_dir.rstrip("/"))
    args.class_name = args.class_name or class_from_dir(class_dir)
    if args.class_name not in OBJECTS:
        ap.error(f"Unknown class {args.class_name!r}; expected one of {sorted(OBJECTS)}")

    images = read_failed(args.retry_from) if args.retry_from else scan_images(args.image_dir)
    if not images:
        raise SystemExit(f"No images to process in {args.retry_from or args.image_dir}")

    client = build_client(args)

    save_root = os.path.join(args.out_root, class_dir, str(args.n), args.semantic)
    os.makedirs(save_root, exist_ok=True)

    short = short_name(args.class_name)
    failed_log = os.path.join(args.out_root, args.failed or f"failed-{short}-{args.n}-{args.semantic}.jsonl")

    with jsonlines.open(failed_log, mode="a") as writer:
        run(client, args, images, save_root, writer)
    print(f"Failure log: {failed_log}")


if __name__ == "__main__":
    main()
