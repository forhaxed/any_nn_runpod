"""
Configuration: two files, and each one is somewhere deliberate.

``remote/anr.toml`` is the **recipe**.  It describes the environment the
training script needs and which pod can host it, and it lives in ``remote/``
because it travels with the code it describes -- the same file builds the venv
on your machine for ``run.py local`` and on the pod for ``run.py start``.

``anr.toml`` in the project root is the **project**: where ``local/`` and
``remote/`` are, and what to do with the pod when a run ends.  It stays here
and is never uploaded.

Both are optional.  With neither, the defaults describe exactly the layout the
launcher creates.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field

PROJECT_FILE = "anr.toml"
RECIPE_FILE = "anr.toml"

#: What ends a run's pod, when there is one.
ON_FINISH = ("terminate", "stop", "keep")


@dataclass
class Recipe:
    """``remote/anr.toml`` -- the environment, and the pod that can host it."""

    entry: str = "train.py"
    output_dir: str = "out"

    # -- environment ------------------------------------------------
    python: str | None = None
    torch: str | None = None
    torchvision: str | None = None
    torchaudio: str | None = None
    torch_index: str | None = None
    requirements: list = field(default_factory=list)

    # -- pod --------------------------------------------------------
    image: str = "runpod/pytorch:1.1.0-cu1290-torch291-ubuntu2204"
    gpu: list = field(default_factory=list)
    gpu_count: int = 1
    container_disk_gb: int = 60
    volume_gb: int = 0
    network_volume_id: str | None = None
    cloud_type: str = "SECURE"
    data_centers: list = field(default_factory=list)
    env: dict = field(default_factory=dict)

    @property
    def environment(self) -> dict:
        """Just the part that decides what a venv contains.

        Kept separate because it is what the cache is keyed on: changing the
        GPU you rent must not rebuild the environment.
        """
        return {
            "python": self.python,
            "torch": self.torch,
            "torchvision": self.torchvision,
            "torchaudio": self.torchaudio,
            "torch_index": self.torch_index,
            "requirements": sorted(self.requirements),
        }

    @classmethod
    def load(cls, remote_dir: str) -> "Recipe":
        data = _read(os.path.join(remote_dir, RECIPE_FILE))
        run = data.get("run", {})
        env = data.get("env", {})
        pod = data.get("pod", {})
        recipe = cls(
            entry=run.get("entry", cls.entry),
            output_dir=run.get("output_dir", cls.output_dir),
            python=env.get("python"),
            torch=env.get("torch"),
            torchvision=env.get("torchvision"),
            torchaudio=env.get("torchaudio"),
            torch_index=env.get("torch_index"),
            requirements=list(env.get("requirements", [])),
            image=pod.get("image", cls.image),
            gpu=list(pod.get("gpu", [])),
            gpu_count=int(pod.get("gpu_count", 1)),
            container_disk_gb=int(pod.get("container_disk_gb", 60)),
            volume_gb=int(pod.get("volume_gb", 0)),
            network_volume_id=pod.get("network_volume_id"),
            cloud_type=pod.get("cloud_type", cls.cloud_type),
            data_centers=list(pod.get("data_centers", [])),
            env=dict(pod.get("env", {})),
        )
        entry_path = os.path.join(remote_dir, recipe.entry)
        if not os.path.isfile(entry_path):
            raise FileNotFoundError(
                f"the recipe names {recipe.entry!r} as the entry point, but "
                f"{entry_path} does not exist"
            )
        return recipe


@dataclass
class Project:
    """``anr.toml`` in the project root -- paths and pod policy."""

    root: str = "."
    local_dir: str = "local"
    remote_dir: str = "remote"
    output_dir: str = "local/out"
    pod_name: str = "anr"
    on_finish: str = "terminate"
    max_hours: float = 0.0
    library_source: str = ""
    ports: tuple = (7777, 7778)

    @property
    def local_entry(self) -> str:
        return os.path.join(self.root, self.local_dir, "local.py")

    @property
    def remote_path(self) -> str:
        return os.path.join(self.root, self.remote_dir)

    @property
    def output_path(self) -> str:
        return os.path.join(self.root, self.output_dir)

    @property
    def has_local(self) -> bool:
        return os.path.isfile(self.local_entry)

    @classmethod
    def load(cls, root: str = ".") -> "Project":
        root = os.path.abspath(root)
        data = _read(os.path.join(root, PROJECT_FILE))
        paths = data.get("paths", {})
        pod = data.get("pod", {})
        project = cls(
            root=root,
            local_dir=paths.get("local", cls.local_dir),
            remote_dir=paths.get("remote", cls.remote_dir),
            output_dir=paths.get("output", cls.output_dir),
            pod_name=pod.get("name", cls.pod_name),
            on_finish=pod.get("on_finish", cls.on_finish),
            max_hours=float(pod.get("max_hours", 0)),
            library_source=pod.get("library_source", cls.library_source),
        )
        if project.on_finish not in ON_FINISH:
            raise ValueError(
                f"pod.on_finish must be one of {', '.join(ON_FINISH)}, "
                f"not {project.on_finish!r}"
            )
        return project


def _read(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    with open(path, "rb") as handle:
        return tomllib.load(handle)
