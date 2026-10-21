"""The rules of Assignment 2 that code can check, in one place.

This file is complete. Do not change it: the grader uses the same file, so a
change here only makes your own numbers disagree with the grader's.

Both tracks hand in a model that answers one question: in which of the five
cities was this photo taken? The grader prepares every test photo with
`grader_input` below and feeds the model a batch of them.

    input   "image"   float32, shape (N, 3, S, S), RGB, values from 0 to 1
    output  "logits"  float32, shape (N, 5), in the order of CITIES

S is 224 in Track A. In Track B you choose it, and write it in
submission.json. The model does its own normalisation; export.py puts it
inside the ONNX file.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

CITIES = ["seoul", "busan", "tokyo", "osaka", "taipei"]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Track A: the same model, the same data, the same number of training photos.
TRACK_A_IMAGE_SIZE = 224
TRACK_A_BUDGET = 60_000          # photos through the network while training
TRACK_A_BATCH_SIZES = (16, 256)  # smallest and largest batch a recipe may use
TRACK_A_SEEDS = (0, 1, 2)        # the grader trains three times and averages

# Track B: any data and any training, but the model has to stay small.
TRACK_B_MAX_PARAMETERS = 12_000_000
TRACK_B_MAX_GMACS = 2.0          # billions of multiply-adds for one photo
TRACK_B_IMAGE_SIZES = (96, 512)  # smallest and largest S
TRACK_B_MAX_FILE_MB = 95         # GitHub refuses files over 100 MB

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def load_photo(path):
    """Open a photo the way the grader does: turned upright, in RGB.

    Phone cameras store a photo sideways and add a note saying which way is
    up. exif_transpose applies that note, so a portrait photo comes out as a
    portrait.
    """
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")


def grader_input(image, size):
    """What the model is given for one photo: a (3, size, size) float32 array.

    The shorter side is resized to `size` (bilinear, as torchvision's
    Resize does for a PIL image), the middle square is cut out, and the values
    are scaled to 0..1. Nothing else: no normalisation.
    """
    width, height = image.size
    if width <= height:
        new_w, new_h = size, int(size * height / width)
    else:
        new_w, new_h = int(size * width / height), size
    image = image.resize((new_w, new_h), Image.BILINEAR)
    left = int(round((new_w - size) / 2.0))
    top = int(round((new_h - size) / 2.0))
    image = image.crop((left, top, left + size, top + size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array.transpose(2, 0, 1).copy()


def list_photos(folder):
    """(path, city) for every photo in folder/<city>/, sorted, in CITIES order."""
    folder = Path(folder)
    rows = []
    for city in CITIES:
        city_dir = folder / city
        if not city_dir.is_dir():
            continue
        for path in sorted(city_dir.iterdir()):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                rows.append((path, city))
    return rows


def balanced_accuracy(true, pred):
    """The score: accuracy in each city, averaged over the five cities.

    The leaderboard and final photos have the same number of photos for
    every city, so this is the same as plain accuracy there. On the public validation set the
    cities have different numbers of photos, and this keeps Busan, the
    smallest, from counting for less.
    """
    true, pred = np.asarray(true), np.asarray(pred)
    confusion = np.zeros((len(CITIES), len(CITIES)), dtype=int)
    for t, p in zip(true, pred):
        confusion[t, p] += 1
    per_city = {}
    for i, city in enumerate(CITIES):
        if confusion[i].sum():
            per_city[city] = float(confusion[i, i] / confusion[i].sum())
    score = float(np.mean(list(per_city.values()))) if per_city else 0.0
    return score, per_city, confusion


# ---------------------------------------------------------------------------
# Measuring an ONNX file: parameters and multiply-adds

def _shape_of(name, shapes, initializers):
    if name in initializers:
        return list(initializers[name].dims)
    return shapes.get(name)


def measure_onnx(path, size):
    """Parameters and multiply-adds of an ONNX model for one size x size photo.

    Parameters are all numbers stored in the file (initializers and constant
    tensors). Multiply-adds are counted for Conv, ConvTranspose, Gemm and
    MatMul, which is where almost all the work in an image model is; the
    elementwise operations around them are left out, as is usual.

    Returns a dict with parameters, gmacs, file_mb, opset, ops and problems,
    a list of the rules the file breaks (empty when it is fine).
    """
    import onnx
    from onnx import numpy_helper, shape_inference

    path = Path(path)
    model = onnx.load(str(path))
    problems = []

    graph = model.graph
    real_inputs = [i for i in graph.input if i.name not in {t.name for t in graph.initializer}]
    if [i.name for i in real_inputs] != ["image"]:
        problems.append(f"the model must have one input called 'image', it has {[i.name for i in real_inputs]}")
    if [o.name for o in graph.output] != ["logits"]:
        problems.append(f"the model must have one output called 'logits', it has {[o.name for o in graph.output]}")

    # Fix the input to one photo of the declared size, then let ONNX work out
    # the shape of everything else.
    for inp in real_inputs:
        dims = inp.type.tensor_type.shape.dim
        if len(dims) != 4:
            problems.append(f"'image' must have 4 dimensions (N, 3, S, S), it has {len(dims)}")
            continue
        fixed = [d.dim_value if d.HasField("dim_value") else None for d in dims]
        if any(f is not None and f != want for f, want in zip(fixed[1:], (3, size, size))):
            problems.append(f"'image' has the shape {fixed}, not (N, 3, {size}, {size})")
        for d, value in zip(dims, (1, 3, size, size)):
            d.ClearField("dim_param")
            d.dim_value = value
    inferred = shape_inference.infer_shapes(model)

    shapes = {}
    for info in list(inferred.graph.value_info) + list(inferred.graph.input) + list(inferred.graph.output):
        dims = info.type.tensor_type.shape.dim
        if all(d.HasField("dim_value") for d in dims):
            shapes[info.name] = [d.dim_value for d in dims]
    initializers = {t.name: t for t in graph.initializer}

    parameters = sum(int(np.prod(t.dims)) for t in graph.initializer)
    for node in graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    parameters += int(numpy_helper.to_array(attr.t).size)

    ops = sorted({n.op_type for n in graph.node})
    for node in graph.node:
        if node.domain not in ("", "ai.onnx"):
            problems.append(f"operation {node.op_type} from domain {node.domain!r} is not standard ONNX")
        if node.op_type in ("Loop", "If", "Scan"):
            problems.append(f"operation {node.op_type} (control flow) is not allowed")

    macs = 0
    unknown = []
    for node in inferred.graph.node:
        if node.op_type in ("Conv", "ConvTranspose"):
            weight = _shape_of(node.input[1], shapes, initializers)
            out = shapes.get(node.output[0])
            inp = shapes.get(node.input[0])
            if weight is None or out is None or inp is None:
                unknown.append(node.name or node.op_type)
                continue
            if node.op_type == "Conv":
                # out: (1, Cout, H, W); weight: (Cout, Cin/groups, kh, kw)
                macs += int(np.prod(out)) * int(np.prod(weight[1:]))
            else:
                # weight: (Cin, Cout/groups, kh, kw); every input pixel is spread over a kernel
                macs += int(np.prod(inp)) * int(np.prod(weight[1:]))
        elif node.op_type in ("Gemm", "MatMul"):
            a = _shape_of(node.input[0], shapes, initializers)
            b = _shape_of(node.input[1], shapes, initializers)
            out = shapes.get(node.output[0])
            if a is None or b is None or out is None:
                unknown.append(node.name or node.op_type)
                continue
            if node.op_type == "Gemm":
                trans_a = any(at.name == "transA" and at.i for at in node.attribute)
                k = a[0] if trans_a else a[-1]
            else:
                k = a[-1]
            macs += int(np.prod(out)) * int(k)
    if unknown:
        problems.append(f"could not work out the size of {len(unknown)} operations ({', '.join(unknown[:3])}); "
                        "export with a fixed image size")

    return {
        "parameters": parameters,
        "gmacs": macs / 1e9,
        "file_mb": path.stat().st_size / 1e6,
        "opset": max((o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), default=0),
        "ops": ops,
        "problems": problems,
    }
