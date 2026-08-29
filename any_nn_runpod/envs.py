"""
Building the environment a run asks for.

The recipe lives in ``remote/anr.toml`` and travels with the code, so the same
file describes the environment here and on the pod.  ``uv`` does the work; it
will download a standalone CPython if the machine has no suitable one.

Two shortcuts matter in practice:

* if the interpreter already running satisfies the recipe, it is used as-is.
  Locally that is almost always true, and it turns a 2.5 GB torch download into
  nothing at all.
* otherwise the venv is cached under a hash of the recipe, so building it is a
  once-per-recipe cost rather than a per-run one.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

DEFAULT_CACHE = os.path.join(os.path.expanduser("~"), ".anr", "envs")

_READY = ".anr_ready"


def spec_key(recipe: dict) -> str:
    return hashlib.sha256(
        json.dumps(recipe, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def series(version: str) -> str:
    return ".".join(version.split("+")[0].split(".")[:2])


def satisfied_by_current(recipe: dict) -> bool:
    """True when the interpreter already running is good enough."""
    import platform

    wanted_python = recipe.get("python")
    if wanted_python and series(platform.python_version()) != series(wanted_python):
        return False

    wanted_torch = recipe.get("torch")
    if wanted_torch:
        try:
            import torch
        except ImportError:
            return False
        if series(torch.__version__) != series(wanted_torch):
            return False

    for requirement in recipe.get("requirements") or []:
        if not _importable(requirement):
            return False
    return True


def _importable(requirement: str) -> bool:
    """Is this requirement already satisfiable here?  Name only, not version.

    A version-pinned requirement is never assumed satisfied: pins exist because
    the exact version matters, and guessing wrong is worse than a rebuild.
    """
    import importlib.util
    import re

    if re.search(r"[<>=!~@]", requirement) or requirement.startswith(("git+", ".", "/")):
        return False
    name = re.split(r"[\[;\s]", requirement.strip())[0].replace("-", "_")
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def resolve(
    recipe: dict,
    cache_root: str = DEFAULT_CACHE,
    install: list | None = None,
    say=print,
    force: bool = False,
) -> str:
    """Return a python executable that satisfies ``recipe``, building if needed.

    ``install`` is extra packages to put in it that are not part of the recipe
    hash -- any_nn_runpod itself, typically, which should track whatever is
    checked out rather than pinning the venv to a version.
    """
    if not force and satisfied_by_current(recipe):
        say(
            f"Environment: using this interpreter (python "
            f"{series(sys.version.split()[0])}"
            + (f", torch {recipe['torch']}" if recipe.get("torch") else "")
            + ") -- it already matches the recipe."
        )
        return sys.executable

    directory = os.path.join(cache_root, spec_key(recipe))
    python = venv_python(directory)
    marker = os.path.join(directory, _READY)

    if not force and os.path.exists(marker) and os.path.exists(python):
        say(f"Environment: reusing {directory}")
        _install_extras(python, install, say, quiet=True)
        return python

    say(f"Environment: building for {_describe(recipe)} -- this happens once.")
    started = time.perf_counter()
    shutil.rmtree(directory, ignore_errors=True)
    os.makedirs(os.path.dirname(directory) or ".", exist_ok=True)

    _ensure_uv(say)
    uv = [sys.executable, "-m", "uv"]
    create = uv + ["venv", directory]
    if recipe.get("python"):
        create += ["--python", str(recipe["python"])]
    _run(create, say, "creating the venv")

    torch_packages = []
    if recipe.get("torch"):
        torch_packages.append(f"torch=={recipe['torch']}")
    if recipe.get("torchvision"):
        torch_packages.append(f"torchvision=={recipe['torchvision']}")
    if recipe.get("torchaudio"):
        torch_packages.append(f"torchaudio=={recipe['torchaudio']}")
    if torch_packages:
        # The pytorch index carries only torch's own wheels, so it gets its own
        # install step -- pointing everything at it would fail to find the rest.
        command = uv + ["pip", "install", "--python", python, *torch_packages]
        if recipe.get("torch_index"):
            command += ["--index-url", str(recipe["torch_index"])]
        _run(command, say, "installing torch")

    requirements = list(recipe.get("requirements") or [])
    if requirements:
        _run(
            uv + ["pip", "install", "--python", python, *requirements],
            say,
            "installing the run's packages",
        )

    _install_extras(python, install, say)

    os.makedirs(directory, exist_ok=True)
    with open(marker, "w") as handle:
        json.dump(recipe, handle, indent=2, default=str)
    say(f"Environment ready in {time.perf_counter() - started:.0f}s: {directory}")
    return python


def venv_python(directory: str) -> str:
    if os.name == "nt":
        return os.path.join(directory, "Scripts", "python.exe")
    return os.path.join(directory, "bin", "python")


def _describe(recipe: dict) -> str:
    bits = [f"python {recipe.get('python', 'any')}"]
    if recipe.get("torch"):
        bits.append(f"torch {recipe['torch']}")
    count = len(recipe.get("requirements") or [])
    if count:
        bits.append(f"{count} package{'s' if count != 1 else ''}")
    return ", ".join(bits)


def _install_extras(python, install, say, quiet=False):
    """Put any_nn_runpod (and friends) in, always fresh.

    Outside the recipe hash on purpose: the venv should follow whatever is
    checked out, not pin itself to the version that first built it.
    """
    if not install:
        return
    if python == sys.executable:
        return  # already running out of it
    _run(
        [sys.executable, "-m", "uv", "pip", "install", "--python", python, *install],
        say,
        "installing any_nn_runpod",
        quiet=quiet,
    )


def _ensure_uv(say):
    try:
        subprocess.run(
            [sys.executable, "-m", "uv", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return
    except (subprocess.CalledProcessError, OSError):
        pass
    _run([sys.executable, "-m", "pip", "install", "--no-input", "uv"], say, "installing uv")


def _run(command, say, what, quiet=False, quiet_for=2.0):
    """Run a build step, reporting as it goes.

    Downloading torch takes minutes.  Collecting the output and printing it at
    the end would leave you staring at a dead terminal, unable to tell a slow
    download from a hung machine -- so lines come out while the command runs,
    throttled because uv redraws progress far faster than anyone can read.
    """
    if not quiet:
        say(f"  {what}...")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        errors="replace",
    )

    tail = []
    last_spoke = 0.0
    for line in process.stdout:
        line = line.strip()
        if not line:
            continue
        tail.append(line)
        del tail[:-40]  # enough context if this ends up in an error
        now = time.monotonic()
        if not quiet and now - last_spoke >= quiet_for:
            say(f"  {line}")
            last_spoke = now

    process.wait()
    if process.returncode != 0:
        raise RuntimeError(
            f"failed while {what} (exit {process.returncode}):\n" + "\n".join(tail)
        )
    if tail and not quiet:
        say(f"  {tail[-1]}")
