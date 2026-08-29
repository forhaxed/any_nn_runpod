"""
The training side of the CIFAR example.

Nothing in this file knows whether it is running on your machine or on a rented
GPU, and nothing in it knows whether the batches come over a wire or out of a
local dataloader.  That is the whole design: one script, three ways to run it.

    python run.py local              batches stream from local/local.py
    python run.py local --no-link    the fallback below builds them here
    python run.py start              the same, on a pod
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from any_nn_runpod import DatasetWrapper, RunpodTrainer, session

BATCH_SIZE = 64
EPOCHS = 2
LEARNING_RATE = 3e-3
GRADIENT_ACCUMULATION_STEPS = 2

#: Batches kept ready on this machine at all times.  The local side tops the
#: window back up as they are consumed, so the GPU never waits for the network.
PRECACHE = 48
#: Batches packed into one message.  Amortizes per-message overhead.
PREPARE = 8


class SmallCNN(nn.Module):
    def __init__(self, classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, classes)

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


class CifarTrainer(RunpodTrainer):
    def train_step(self, step, batch, device, weight_dtype):
        images, labels = batch
        images = images.to(device, dtype=torch.float32, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = self.models[0](images)
        loss = F.cross_entropy(logits, labels)
        accuracy = (logits.argmax(1) == labels).float().mean()
        return loss, {"accuracy": accuracy}

    def eval_step(self, step, batch, device, weight_dtype):
        return self.train_step(step, batch, device, weight_dtype)

    def eval_end(self, step):
        """Send a handful of predictions home to be drawn.

        The pictures are made where the class names live; this side only ships
        the tensors.  Without a local side this is quietly dropped, and the run
        carries on.
        """
        batch = next(iter(self.eval_dataloader), None)
        if batch is None:
            return
        images, labels = batch
        with torch.no_grad():
            predicted = self.models[0](
                images[:8].to(self.device, dtype=torch.float32)
            ).argmax(1)
        self.send_artifact(
            "samples",
            {
                "images": images[:8].cpu(),
                "predicted": predicted.cpu(),
                "actual": labels[:8].cpu(),
            },
            step,
        )


def unpack(batches):
    """Runs here, with the GPU available.  Undoes the local side's fp16 cast."""
    return [(images.float(), labels) for images, labels in batches]


def local_cifar(train: bool):
    """Used only when there is no local side: build the data right here."""
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )
    return DataLoader(
        datasets.CIFAR10(
            session.path("data"), train=train, download=True, transform=transform
        ),
        batch_size=BATCH_SIZE,
        shuffle=train,
        drop_last=True,
    )


def main():
    torch.backends.cudnn.benchmark = True

    model = SmallCNN()
    trainer = CifarTrainer(output_dir=session.output_dir)
    trainer.models = [model]
    trainer.optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    trainer.train_dataloader = DatasetWrapper(
        "train",
        precache=PRECACHE,
        prepare=PREPARE,
        unpack=unpack,
        fallback=lambda: local_cifar(train=True),
    )
    trainer.eval_dataloader = DatasetWrapper(
        "eval",
        precache=16,
        prepare=PREPARE,
        unpack=unpack,
        optional=True,
        fallback=lambda: local_cifar(train=False),
    )

    trainer.batch_size = BATCH_SIZE
    trainer.epochs = EPOCHS
    trainer.gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS
    trainer.mixed_precision = "bf16" if torch.cuda.is_available() else "no"
    trainer.max_grad_norm = 1.0
    trainer.seed = 0
    trainer.eval_every_steps = 100
    trainer.save_checkpoint_every_steps = 200

    session.bind(trainer)
    trainer.init()

    # Proof the round trip works in the other direction too: this side asking
    # the local side for something only it has.
    if session.connected:
        answer = session.call("class_names", timeout=30)
        trainer.print(f"Classes, according to the local side: {answer['names']}")

    trainer.train()


if __name__ == "__main__":
    main()
