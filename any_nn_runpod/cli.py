"""
The launcher.

    python run.py local            build the environment here, train here
    python run.py local --no-link  the same, with no local side at all

Both do exactly what ``run.py start`` will do on a pod, minus renting one: the
same recipe builds the same venv, the same session process runs the same entry
script, and the same link carries the same traffic -- over loopback instead of
a public IP.  Which is the point: almost everything that can go wrong on a pod
goes wrong here first, for free.
"""

from __future__ import annotations

import argparse
import os
import runpy
import socket
import subprocess
import sys
import threading
import time

from colorama import Fore, Style, init as _init_colorama

from any_nn_runpod import envs
from any_nn_runpod.manifest import Project, Recipe

_init_colorama()


def _say(text=""):
    print(text, flush=True)


def _heading(text):
    _say(f"\n{Fore.CYAN}{text}{Style.RESET_ALL}")


# ======================================================================
#  local
# ======================================================================
def command_local(project: Project, args) -> int:
    recipe = Recipe.load(project.remote_path)
    entry = os.path.join(project.remote_path, recipe.entry)

    _heading(f"Recipe: {recipe.entry}")
    python = envs.resolve(
        recipe.environment,
        install=_library_install(),
        say=_say,
        force=args.rebuild,
    )

    use_link = not args.no_link and project.has_local
    if not args.no_link and not project.has_local:
        _say(
            f"{Fore.YELLOW}No {project.local_entry} -- running standalone. "
            f"(That is a supported mode, not a problem.){Style.RESET_ALL}"
        )

    port = args.port or _free_port()
    output_dir = (
        os.path.abspath(project.output_path)
        if use_link
        else os.path.join(project.remote_path, recipe.output_dir)
    )

    command = [
        python,
        "-u",
        "-m",
        "any_nn_runpod.run_session",
        "--entry",
        entry,
        "--output-dir",
        output_dir,
    ]
    command += (
        ["--host", "127.0.0.1", "--port", str(port), "--wait", "120"]
        if use_link
        else ["--no-link"]
    )

    _heading("Session")
    _say(f"  {os.path.basename(python)} running {recipe.entry}"
         + (f", listening on 127.0.0.1:{port}" if use_link else " standalone"))

    child = subprocess.Popen(
        command,
        cwd=project.remote_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,  # bytes, so carriage returns survive to _pump_output
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    pump = threading.Thread(
        target=_pump_output, args=(child, "session"), daemon=True
    )
    pump.start()

    if not use_link:
        code = child.wait()
        pump.join(timeout=5)
        _report_exit(code)
        return code

    app = None
    try:
        app = _drive_local(project, "127.0.0.1", port, child)
    finally:
        code = _wind_down(child, app)
    return code


def _drive_local(project: Project, host: str, port: int, child):
    """Load ``local/local.py``, connect it to the session, and wait it out."""
    from any_nn_runpod.link import Link

    app = _load_local_app(project)
    channel = _dial(host, port, child, timeout=120)
    link = Link(channel, role="local").start(initiate=True)

    _heading("Training")
    app.attach(link)
    app.wait()
    _report_run(app)
    return app


def _wind_down(child, app) -> int:
    if app is not None:
        app.shutdown()
        if app.link is not None:
            app.link.close("launcher done")
    try:
        code = child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        _say(f"{Fore.YELLOW}Session did not exit; killing it.{Style.RESET_ALL}")
        child.kill()
        code = child.wait()
    if app is not None and app.failure:
        return 1
    _report_exit(code)
    return code


def _report_exit(code):
    if code:
        _say(f"{Fore.RED}Session exited with code {code}.{Style.RESET_ALL}")


def _load_local_app(project: Project):
    """Import ``local/local.py`` and pick the ``Local`` app out of it.

    Loaded with the working directory set to its own folder, mirroring the
    session -- which runs with the working directory set to ``remote/``.  So
    each side's relative paths mean "next to me", whichever directory ``run.py``
    itself was invoked from.
    """
    from any_nn_runpod.local import Local

    path = project.local_entry
    directory = os.path.dirname(os.path.abspath(path))
    sys.path.insert(0, directory)
    os.chdir(directory)
    namespace = runpy.run_path(path, run_name="anr_local")
    for value in namespace.values():
        if isinstance(value, Local):
            return value
    raise SystemExit(
        f"{path} does not define a Local app.\n"
        "Add one at module level:\n\n"
        "    from any_nn_runpod import Local\n"
        "    app = Local(output_dir='out')\n"
    )


def _dial(host, port, child, timeout: float):
    """Connect to the session, giving it time to come up -- or say why it didn't."""
    from any_nn_runpod.wire.protocol import Channel
    from any_nn_runpod.wire.transport import TcpTransport

    deadline = time.monotonic() + timeout
    while True:
        if child.poll() is not None:
            raise SystemExit(
                f"The session exited with code {child.returncode} before the "
                "local side could connect. Its output is above."
            )
        try:
            return Channel(TcpTransport.connect(host, port, timeout=5))
        except OSError as exc:
            if time.monotonic() >= deadline:
                raise SystemExit(
                    f"could not reach the session on {host}:{port} within "
                    f"{timeout:.0f}s: {exc}"
                ) from exc
            time.sleep(0.25)


def _pump_output(child, label):
    """Mirror the session's output, keeping progress bars on one line.

    The session's own children -- pip, uv, a dataset download -- draw with
    carriage returns.  Reading in text mode would turn every redraw into a
    separate line and bury the actual output under thousands of them, so the
    stream is read as bytes and ``\\r`` is honoured as what it means: rewrite
    the line you are on.
    """
    prefix = f"{Fore.BLACK}{Style.BRIGHT}[{label}]{Style.RESET_ALL} "
    pending = bytearray()
    transient = False   # the line on screen is a redraw, not a finished line
    after_cr = False    # last byte was \r; \n next would make it a CRLF pair

    def emit(raw: bytes, overwrite: bool):
        nonlocal transient
        text = raw.decode("utf-8", "replace")
        if not text.strip() and not overwrite:
            if transient:  # a blank line after a redraw: finish that line off
                sys.stdout.write("\n")
                sys.stdout.flush()
                transient = False
            return
        # Overwriting needs padding, or the tail of a longer previous redraw
        # stays on screen behind the shorter new one.
        line = ("\r" if transient else "") + prefix + text
        sys.stdout.write(line + ("        \r" if overwrite else "\n"))
        sys.stdout.flush()
        transient = overwrite

    while True:
        # Unbuffered, so this is one read() returning whatever has arrived --
        # not a wait for 4096 bytes that would stall a quiet training loop.
        chunk = child.stdout.read(4096)
        if not chunk:
            break
        for byte in chunk:
            if after_cr:
                after_cr = False
                if byte == 10:
                    # CRLF: one line ending, not a redraw. Windows text-mode
                    # stdout writes every newline this way, so treating the \r
                    # as a redraw would double every line the session prints.
                    emit(bytes(pending), overwrite=False)
                    pending.clear()
                    continue
                emit(bytes(pending), overwrite=True)
                pending.clear()
            if byte == 13:
                after_cr = True
            elif byte == 10:
                emit(bytes(pending), overwrite=False)
                pending.clear()
            else:
                pending.append(byte)

    if after_cr:
        emit(bytes(pending), overwrite=True)
    elif pending:
        emit(bytes(pending), overwrite=False)
    if transient:
        sys.stdout.write("\n")
        sys.stdout.flush()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _library_install() -> list:
    """How to put any_nn_runpod into a freshly built venv.

    An editable checkout installs itself so the venv follows your edits; a
    pip-installed copy installs the same version by name.
    """
    import any_nn_runpod

    root = os.path.dirname(os.path.dirname(os.path.abspath(any_nn_runpod.__file__)))
    if os.path.isfile(os.path.join(root, "pyproject.toml")):
        return ["-e", root]
    return [f"any_nn_runpod=={any_nn_runpod.__version__}"]


# ======================================================================
#  pod commands
# ======================================================================
def _apply_overrides(recipe, args):
    """Command-line overrides for the pod half of the recipe.

    Several GPU ids are worth naming: RunPod picks whichever of them is free,
    and "no instances currently available" is the single most common way a
    create fails.
    """
    gpus = None
    if getattr(args, "gpu", None):
        gpus = [name.strip() for name in args.gpu.split(",") if name.strip()]
    if getattr(args, "cloud", None):
        recipe.cloud_type = args.cloud
    return gpus


def _client(project):
    from any_nn_runpod.cloud.api import RunPod

    return RunPod(root=project.root)


def command_gpus(project, args) -> int:
    """What is available and what it costs -- the menu for --gpu."""
    types = _client(project).gpu_types()
    usable = [gpu for gpu in types if gpu["creatable"]]
    hidden = len(types) - len(usable)
    _heading(f"{len(usable)} GPU types you can create a pod with")
    _say(f"  {'$/hr':>7}  {'GPU':<34} {'VRAM':>6}  clouds")
    for gpu in usable[: args.limit]:
        clouds = ", ".join(
            [name for name, on in (("secure", gpu["secure"]), ("community", gpu["community"])) if on]
        )
        _say(
            f"  {gpu['price']:>7.3f}  {gpu['id']:<34} "
            f"{(gpu['memory_gb'] or 0):>4}GB  {clouds}"
        )
    if hidden:
        _say(
            f"\n({hidden} more exist in RunPod's catalogue but its create "
            "endpoint will not accept them, so they are not listed.)"
        )
    _say(
        "\nName several, not one: [pod] gpu = [\"a\", \"b\"] in remote/anr.toml, "
        "or --gpu 'a,b'. RunPod takes whichever is free, and \"no instances "
        "currently available\" is the usual answer to a single choice."
    )
    return 0


def command_ps(project, args) -> int:
    """List the pods this tool created.  Other pods are counted, never listed."""
    from any_nn_runpod.cloud import driver, lifecycle

    ours, others = lifecycle.managed_pods(_client(project))
    _heading(f"{len(ours)} pod(s) created by any_nn_runpod")
    for pod in ours:
        _say("  " + driver.describe(pod))
    if not ours:
        _say("  (none)")
    if others:
        _say(
            f"\n{others} other pod(s) on this account are not shown: they were "
            "not created here, and this tool will not touch them."
        )
    return 0


def command_down(project, args) -> int:
    from any_nn_runpod.cloud import driver, lifecycle

    client = _client(project)
    ours, others = lifecycle.managed_pods(client)
    if not args.all:
        ours = [pod for pod in ours if pod.get("name") == project.pod_name]

    if not ours:
        _say("Nothing to shut down.")
        if others:
            _say(f"({others} pod(s) on this account were not created here.)")
        return 0

    _heading(f"About to {args.action} {len(ours)} pod(s)")
    for pod in ours:
        _say("  " + driver.describe(pod))
    if not args.yes and not _confirm(f"{args.action.capitalize()} these?"):
        _say("Left alone.")
        return 0

    failed = []
    for pod in ours:
        try:
            lifecycle.finish(
                client, pod["id"], args.action, say=lambda t: _say("  " + t)
            )
        except Exception as exc:  # noqa: BLE001
            # Keep going. This is the command someone runs when they want the
            # billing to stop, and stopping at the first awkward pod would
            # leave the rest running for no reason.
            failed.append((pod["id"], exc))
            _say(f"  {Fore.RED}{pod['id']}: {exc}{Style.RESET_ALL}")

    if failed:
        _say(
            f"\n{Fore.RED}{len(failed)} pod(s) could not be ended and are still "
            f"billing. Try again, or use the RunPod console.{Style.RESET_ALL}"
        )
        return 1
    return 0


def command_up(project, args) -> int:
    from any_nn_runpod.cloud import lifecycle

    recipe = Recipe.load(project.remote_path)
    pod = lifecycle.find_or_create(
        _client(project),
        project,
        recipe,
        token=os.environ.get("ANR_TOKEN"),
        gpu_types=[args.gpu] if args.gpu else None,
        say=_say,
    )
    _say(f"\nPod {pod['id']} is up. `run.py start` will sync and train on it.")
    return 0


def command_start(project, args) -> int:
    """The pod path: create/reuse, sync, build, run, attach, wind down."""
    from any_nn_runpod.cloud import driver, lifecycle

    recipe = Recipe.load(project.remote_path)
    client = _client(project)
    token = os.environ.get("ANR_TOKEN")

    pod = lifecycle.find_or_create(
        client, project, recipe, token=token,
        gpu_types=_apply_overrides(recipe, args), say=_say,
    )
    deadline = lifecycle.Deadline(project.max_hours)

    control = None
    app = None
    # "failed" until something says otherwise. The pod is ended in the finally
    # either way, but the outcome is also what gets *reported* -- and starting
    # from an optimistic value meant a run that died connecting announced
    # itself as "Run kept", which is a lie about the one thing you were
    # watching for.
    outcome = "failed"
    try:
        control = driver.connect_supervisor(pod, project.ports[0], token=token, say=_say)

        _heading("Workspace")
        _say(f"  remote/ is {driver.workspace_hint(project.remote_path)}")
        driver.push_remote(control, project.remote_path, say=_say)

        _heading("Environment")
        python = driver.build_environment(
            control, recipe, project.library_source, force=args.rebuild, say=_say
        )
        _say(f"  python: {python}")

        use_link = not args.no_link and project.has_local
        if use_link:
            app = _load_local_app(project)

        _heading("Session")
        driver.relay_output(control, _ConsoleRelay(), deadline)
        driver.start_session(control, recipe, python, no_link=not use_link, say=_say)

        if not use_link:
            _say("  running standalone on the pod; ctrl-c to detach")
            outcome = _wait_standalone(control, deadline)
        else:
            driver.attach_local(app, pod, project.ports[1], say=_say)
            _heading("Training")
            outcome = _wait_attached(app, control, deadline)
            _report_run(app)
    except KeyboardInterrupt:
        _say(f"\n{Fore.YELLOW}Interrupted.{Style.RESET_ALL}")
        outcome = "interrupted"
    finally:
        if app is not None:
            app.shutdown()
            if app.link is not None:
                app.link.close("launcher done")
        code = _finish_pod(client, pod, project, outcome, args, deadline)
        if control is not None:
            control.close("launcher done")
    return code


def _wait_attached(app, control, deadline) -> str:
    """Wait out an attached run, but not forever.

    ``app.wait()`` alone blocks until the run ends or the link dies, which
    means a pod that wedges keeps billing until somebody notices. So the wait
    happens in slices with ``max_hours`` and the idle timeout checked between
    them -- the whole reason those settings exist.

    Liveness is measured in bytes over the link, not in messages: a run
    uploading a large checkpoint is silent for minutes and perfectly healthy.
    """
    seen = app.traffic()
    while True:
        if app.wait(timeout=15.0) or app.failure:
            return "failed" if app.failure else "finished"
        if app.link is None or not app.link.connected:
            return "failed" if app.failure else "finished"

        moved = app.traffic()
        if moved != seen:
            seen = moved
            deadline.saw_progress()

        expired = deadline.expired()
        if expired:
            _say(f"\n{Fore.YELLOW}Ending the run: {expired}.{Style.RESET_ALL}")
            try:
                control.call("run.stop", timeout=60)
            except Exception as exc:  # noqa: BLE001 -- the pod goes either way
                _say(f"  (could not stop the session cleanly: {exc})")
            return "expired"


def _wait_standalone(control, deadline) -> str:
    """No local side: just relay output until the session exits or time runs out."""
    exited = {}
    control.handle("session.exit", lambda payload: exited.update(payload or {}))
    while not exited and control.connected:
        expired = deadline.expired()
        if expired:
            _say(f"{Fore.YELLOW}Ending the run: {expired}.{Style.RESET_ALL}")
            control.call("run.stop", timeout=60)
            return "expired"
        time.sleep(1.0)
    return "finished" if exited.get("code") == 0 else "failed"


def _finish_pod(client, pod, project, outcome, args, deadline) -> int:
    from any_nn_runpod.cloud import lifecycle

    policy = args.on_finish or project.on_finish
    _heading(f"Run {outcome} after {deadline.elapsed_hours:.2f}h")

    if outcome == "interrupted" and not args.yes:
        policy = _ask_policy(pod, policy)
    try:
        lifecycle.finish(client, pod["id"], policy, say=lambda t: _say("  " + t))
    except Exception as exc:  # noqa: BLE001 -- never mask the run's own outcome
        _say(f"{Fore.RED}Could not {policy} pod {pod['id']}: {exc}{Style.RESET_ALL}")
        _say(f"{Fore.RED}It is still running and still billing. "
             f"Use `run.py down` or the RunPod console.{Style.RESET_ALL}")
    return 1 if outcome in ("failed", "expired") else 0


def _ask_policy(pod, default: str) -> str:
    _say(f"\nPod {pod['id']} is still up. What should happen to it?")
    _say("  [t] terminate -- destroy it, stop all charges")
    _say("  [s] stop      -- keep the disk (still charged for storage)")
    _say("  [k] keep      -- leave it running (charged in full)")
    try:
        answer = input(f"  choice [{default[0]}]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return default
    return {"t": "terminate", "s": "stop", "k": "keep"}.get(answer, default)


def _confirm(question: str) -> bool:
    try:
        return input(f"{question} [y/N]: ").strip().lower() in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


def _report_run(app):
    if app.failure:
        # The reason, not just the verdict. A run that reports "failed" and
        # then declines to say why is the most annoying possible output, and it
        # is exactly what this printed before.
        _say(f"{Fore.RED}Run failed: {app.failure}{Style.RESET_ALL}")
    elif app.result:
        _say(
            f"\n{Fore.GREEN}Finished at step {app.result.get('global_step')}."
            f"{Style.RESET_ALL}"
        )
        _say(f"tensorboard --logdir {app.result.get('log_dir')}")


class _ConsoleRelay:
    """Prints the pod's session output with a prefix, like the local session's."""

    def text(self, text: str):
        if text:
            sys.stdout.write(text)
            sys.stdout.flush()


# ======================================================================
#  entry
# ======================================================================
def build_parser():
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Write remote/, ship it, keep the data and the logs at home.",
    )
    parser.add_argument(
        "--root", default=".", help="project directory (holds anr.toml, local/, remote/)"
    )
    subparsers = parser.add_subparsers(dest="command")

    local = subparsers.add_parser(
        "local", help="build the environment here and train here, with no pod"
    )
    local.add_argument(
        "--no-link",
        action="store_true",
        help="do not attach local/local.py -- run remote/ entirely on its own",
    )
    local.add_argument(
        "--rebuild", action="store_true", help="rebuild the environment from scratch"
    )
    local.add_argument("--port", type=int, default=0, help="fixed port, for debugging")
    local.set_defaults(handler=command_local)

    gpus = subparsers.add_parser("gpus", help="list available GPU types and prices")
    gpus.add_argument("--limit", type=int, default=25)
    gpus.set_defaults(handler=command_gpus)

    up = subparsers.add_parser("up", help="create (or resume) this project's pod")
    up.add_argument("--gpu", help="GPU type id(s), comma-separated, overriding the recipe")
    up.add_argument("--cloud", choices=("SECURE", "COMMUNITY"))
    up.set_defaults(handler=command_up)

    start = subparsers.add_parser(
        "start", help="sync remote/, build the environment, and train on the pod"
    )
    start.add_argument("--gpu", help="GPU type id(s), comma-separated, overriding the recipe")
    start.add_argument("--cloud", choices=("SECURE", "COMMUNITY"))
    start.add_argument(
        "--no-link",
        action="store_true",
        help="do not attach local/local.py -- run remote/ entirely on the pod",
    )
    start.add_argument(
        "--rebuild", action="store_true", help="rebuild the pod's environment"
    )
    start.add_argument(
        "--on-finish",
        choices=("terminate", "stop", "keep"),
        help="what to do with the pod when the run ends (default: from anr.toml)",
    )
    start.add_argument(
        "--yes", action="store_true", help="do not ask about the pod on interrupt"
    )
    start.set_defaults(handler=command_start)

    ps = subparsers.add_parser("ps", help="list the pods any_nn_runpod created")
    ps.set_defaults(handler=command_ps)

    down = subparsers.add_parser(
        "down", help="stop or terminate pods -- only ones created here"
    )
    down.add_argument(
        "--all", action="store_true", help="every managed pod, not just this project's"
    )
    down.add_argument(
        "--action", choices=("terminate", "stop"), default="terminate"
    )
    down.add_argument("--yes", action="store_true", help="skip the confirmation")
    down.set_defaults(handler=command_down)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "handler", None):
        parser.print_help()
        return 0

    from any_nn_runpod.cloud.api import RunPodError

    try:
        project = Project.load(args.root)
        return args.handler(project, args) or 0
    except KeyboardInterrupt:
        _say(f"\n{Fore.YELLOW}Interrupted.{Style.RESET_ALL}")
        return 130
    except (RunPodError, ConnectionError, FileNotFoundError, ValueError) as problem:
        # The ways a run legitimately cannot start: no API key, a recipe naming
        # an entry that is not there, a pod that never answered. Each of these
        # already carries a sentence saying what to do about it, and a traceback
        # would only bury it.
        _say(f"\n{Fore.RED}{problem}{Style.RESET_ALL}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
