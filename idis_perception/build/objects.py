#!/usr/bin/env python3
"""Distractor object inventory for Idis-perception (paper Table 3).

  classes     : the 9 coarse classes of the ImageNet-9 original split
  aligned     : OBJECTS[<target class>]           - objects that belong with the target
  conflicting : OBJECTS[<another class>]          - objects that belong with a different class
  irrelevant  : IRRELEVANT                        - shared by every class
  n           : the first n entries of a list, so n=1..4 are nested prefixes
  short names : `label` field and meta filenames drop the modifier (`wheeled vehicle` -> `vehicle`)
"""

OBJECTS = {
    "dog": [
        "dog bone chew toy",
        "dog bowl",
        "tennis ball",
        "kennel",
    ],
    "bird": [
        "birdcage",
        "nest",
        "feather",
        "bird feeder",
    ],
    "wheeled vehicle": [
        "tire",
        "steering wheel",
        "license plate",
        "bumper",
    ],
    "reptile": [
        "terrarium rock",
        "heat lamp",
        "log hideout",
        "shed skin",
    ],
    "carnivore": [
        "toy fang",
        "fake blood stain",
        "fake chunk of meat",
        "toy skeletal animal carcass",
    ],
    "insect": [
        "trash bag",
        "empty net designed for catching insects",
        "fruit peel",
        "flower",
    ],
    "musical instrument": [
        "chalkboard with music notes",
        "music stand",
        "metronome",
        "sheet music",
    ],
    "primate": [
        "patch of jungle foliage",
        "banana",
        "coconut",
        "vine",
    ],
    "fish": [
        "fishing rod",
        "large empty nylon fishing net",
        "life jacket",
        "aquarium coral ornament",
    ],
}

IRRELEVANT = ["umbrella", "clock", "tv", "suitcase"]

SEMANTICS = ("aligned", "conflicting", "irrelevant")

# Short forms used for the `label` field and the meta filenames.
SHORT_NAMES = {
    "wheeled vehicle": "vehicle",
    "musical instrument": "instrument",
}


def class_from_dir(dir_name):
    """Class name from an ImageNet-9 directory: `02_wheeled vehicle` -> `wheeled vehicle`."""
    name = dir_name.strip("/").split("/")[-1]
    head, sep, tail = name.partition("_")
    return tail if sep and head.isdigit() else name


def short_name(class_name):
    """`wheeled vehicle` -> `vehicle`; every other class is its own short name."""
    return SHORT_NAMES.get(class_name, class_name)
