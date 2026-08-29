"""
The local side of the CIFAR example.

This machine owns the data and the output.  It never imports anything from
``remote/`` -- the only thing the two share is the *names*: "train", "eval",
"samples".

Run it with ``python run.py local`` from the project root.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from any_nn_runpod import Local

DATA_ROOT = "./data"
BATCH_SIZE = 64

app = Local(output_dir="out")


# ══════════════════════════════ DATA ══════════════════════════════
# Built lazily -- the factory runs the first time the training side asks, so
# nothing is downloaded or opened by merely importing this file.
def _cifar(train: bool):
    tensorize = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )
    augment = (
        [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
        if train
        else []
    )
    return datasets.CIFAR10(
        DATA_ROOT,
        train=train,
        download=True,
        transform=transforms.Compose(augment + tensorize.transforms),
    )


def pack_train(batches):
    """Halve the traffic: fp16 on the wire, fp32 again on the other side.

    Nothing here is required -- without a pack the batch travels as it is.  It
    is the hook for cutting the payload down, and for CIFAR that is one cast.
    """
    return [(images.half(), labels) for images, labels in batches]


app.dataset(
    "train",
    lambda: DataLoader(
        _cifar(train=True),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        drop_last=True,
    ),
    pack=pack_train,
)

app.dataset(
    "eval",
    lambda: DataLoader(
        _cifar(train=False), batch_size=BATCH_SIZE, shuffle=False, drop_last=True
    ),
    pack=pack_train,
)


# ══════════════════════════ WORK SENT HOME ══════════════════════════
CLASSES = (
    "plane", "car", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
STD = torch.tensor([0.247, 0.243, 0.261]).view(3, 1, 1)


@app.on("samples")
def samples(payload, ctx):
    """Draw the prediction grid here, where the class names and the fonts are.

    The training side sent raw images and predictions; turning them into a
    picture is exactly the kind of work it should not be spending GPU minutes
    on. Every handler looks like this: ``(payload, ctx)``, where ctx can log.
    """
    images = payload["images"].float() * STD + MEAN
    grid = torch.cat(list(images.clamp(0, 1)), dim=2)  # one row, side by side
    ctx.log_image("samples", (grid * 255).to(torch.uint8), ctx.step)

    guessed = [CLASSES[i] for i in payload["predicted"].tolist()]
    truth = [CLASSES[i] for i in payload["actual"].tolist()]
    marks = " ".join(
        f"{g}{'' if g == t else f'!={t}'}" for g, t in zip(guessed, truth)
    )
    ctx.print(f"step {ctx.step}: {marks}")


@app.on("class_names")
def class_names(payload, ctx):
    """A plain request/response: the training side asks, this side answers.

    Pointless for CIFAR -- it proves the round trip works, which is what
    ``link.call`` is for when the thing you need genuinely only exists here.
    """
    return {"names": list(CLASSES)}
