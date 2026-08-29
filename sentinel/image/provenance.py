"""File-level provenance: metadata is evidence, and so is its absence.

This module deliberately reports *facts about the container* rather than
guesses about the pixels. In practice it is often the single most decisive
input: a JPEG carrying a camera make, a lens model and an exposure triple was
produced by a camera, and a file whose only software tag names an image
generator was produced by that generator.

The caveat is stated in the signal itself: metadata is trivially strippable,
and almost every social platform strips it on upload. Missing EXIF is
therefore weak evidence at best, which is exactly why it is scored through the
same calibrated machinery as everything else instead of being special-cased
into a verdict.
"""

from __future__ import annotations

import io
import re

from PIL import ExifTags, Image

from ..types import SYNTHETIC_HIGH, SYNTHETIC_LOW, ModuleReport, Signal

MODULE = "provenance"

#: Software/producer strings that name a known generative pipeline.
GENERATOR_HINTS = re.compile(
    r"(stable[\s_-]?diffusion|midjourney|dall[\s_-]?e|firefly|imagen|flux"
    r"|comfyui|automatic1111|invokeai|novelai|leonardo\.ai|playground"
    r"|nightcafe|craiyon|generated\s+by\s+ai|gan\b)",
    re.IGNORECASE,
)

#: Editors. Not proof of anything on their own, but worth surfacing.
EDITOR_HINTS = re.compile(
    r"(photoshop|lightroom|gimp|affinity|snapseed|facetune|picsart|remini)",
    re.IGNORECASE,
)

_TAGS = {v: k for k, v in ExifTags.TAGS.items()}


def _exif_dict(img: Image.Image) -> dict:
    try:
        raw = img.getexif()
    except Exception:
        return {}
    if not raw:
        return {}
    out = {}
    for tag_id, value in raw.items():
        out[ExifTags.TAGS.get(tag_id, str(tag_id))] = value
    try:
        for tag_id, value in raw.get_ifd(_TAGS["ExifOffset"]).items():
            out[ExifTags.TAGS.get(tag_id, str(tag_id))] = value
    except Exception:
        pass
    return out


def analyse(data: bytes) -> ModuleReport:
    rep = ModuleReport(module=MODULE)

    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:  # pragma: no cover - unreadable input
        rep.error = f"could not parse image container: {exc}"
        rep.signals = [
            Signal.missing("exif_capture_completeness", "Camera metadata completeness", "unreadable file"),
            Signal.missing("generator_tag_present", "Generative-tool tag present", "unreadable file"),
        ]
        return rep

    exif = _exif_dict(img)

    # Fields a camera fills in and a generator almost never does.
    capture_fields = [
        "Make",
        "Model",
        "LensModel",
        "ExposureTime",
        "FNumber",
        "ISOSpeedRatings",
        "FocalLength",
        "DateTimeOriginal",
    ]
    present = [f for f in capture_fields if exif.get(f) not in (None, "", 0)]
    completeness = len(present) / len(capture_fields)

    blob = " ".join(str(v) for v in exif.values())
    software = str(exif.get("Software", "")) + " " + str(exif.get("ProcessingSoftware", ""))
    generator_hit = bool(GENERATOR_HINTS.search(blob) or GENERATOR_HINTS.search(software))
    editor_hit = bool(EDITOR_HINTS.search(blob) or EDITOR_HINTS.search(software))

    # C2PA / Content Credentials arrive as a JUMBF box. Its presence means the
    # file carries a signed provenance manifest worth reading out of band.
    c2pa = b"c2pa" in data[:262144].lower() or b"jumb" in data[:262144].lower()

    rep.signals = [
        Signal(
            key="exif_capture_completeness",
            label="Camera metadata completeness",
            value=round(completeness, 4),
            unit="fraction of capture fields",
            # Rich camera metadata argues for authenticity, so a *low* value
            # is the synthetic-leaning direction.
            direction=SYNTHETIC_LOW,
            context={"fields_present": present, "software": software.strip()},
        ),
        Signal(
            key="generator_tag_present",
            label="Generative-tool tag present",
            value=1.0 if generator_hit else 0.0,
            unit="boolean",
            direction=SYNTHETIC_HIGH,
        ),
    ]

    if generator_hit:
        rep.notes.append(f"Metadata names a generative tool: {software.strip()!r}")
    if editor_hit:
        rep.notes.append(f"Metadata names an image editor: {software.strip()!r}")
    if c2pa:
        rep.notes.append(
            "File carries a C2PA/JUMBF provenance box; verify it with a C2PA "
            "validator for a signed answer that does not depend on pixel analysis."
        )
    if not exif:
        rep.notes.append(
            "No EXIF at all. Common for AI output, but also the normal result "
            "of uploading through any social platform, so weigh it lightly."
        )
    return rep
