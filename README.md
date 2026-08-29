# any_nn_runpod

Train on a rented GPU without giving up your data, your logs or your
checkpoints.

You write three files. One command runs them.

```
remote/train.py   the training script -- written like any_nn
remote/anr.toml   the recipe: what environment it needs, what pod can host it
local/local.py    what your machine offers: datasets, and handlers for work
                  that should finish at home
```

```bash
python run.py local     # build the environment here, train here
python run.py start     # rent a pod, ship remote/, train there
```

The same `remote/train.py` serves both. It never learns which.

---

## Setting it up

**1. Install the library, editable.**

```bash
git clone https://github.com/forhaxed/any_nn_runpod
cd any_nn_runpod && pip install -e .
```

Editable matters: `import any_nn_runpod` then resolves to the checkout, so
edits take effect with no reinstall — new modules included. Only changes to
`pyproject.toml` (dependencies, entry points) need `pip install -e .` again.

**2. Lay out a project.** It can be anywhere; `run.py --root <dir>` points at it.

```
myproject/
  run.py  run.bat  run.sh    copied from the library checkout
  anr.toml                   paths and pod policy      (gitignore it)
  .env                       RUNPOD_API_KEY=...        (gitignore it)
  local/local.py             what this machine offers  (never uploaded)
  remote/anr.toml            the recipe
  remote/train.py            the training script       (uploaded whole)
```

Nothing is mandatory except `remote/` with a recipe and an entry script. With
no `local/local.py` the run is standalone, which is a supported mode.

**3. Get a RunPod API key** from
[console.runpod.io](https://console.runpod.io/user/settings) → API Keys, and
put it in `.env` at the project root:

```
RUNPOD_API_KEY=rpa_...
```

It stays there. It is never put in a pod's environment — see *Pods and money*.

**4. Point the pod at a copy of this library it can reach.** In `anr.toml`:

```toml
[pod]
library_source = "git+https://github.com/YOU/any_nn_runpod.git"
```

This is the one setting that is easy to miss and confusing to debug. A pod
installs the library from git on boot, so **`run.py start` runs whatever is
pushed, not what is in your working tree**. `run.py local` runs your working
tree. When the two disagree, the pod is running the old code:

```bash
git -C <library> status --short           # uncommitted?
git -C <library> diff --stat origin/master  # unpushed?
```

Edits to `remote/*` are different — those are uploaded on every `run.py start`
and need no commit.

**5. Check it works before renting anything.**

```bash
python run.py local
```

That builds the environment the recipe asks for, runs the entry script, and
attaches `local/local.py` — the same steps `start` takes on a pod, over
loopback. Almost everything that can go wrong on a pod goes wrong here first,
for free.

## Commands

```
run.py local  [--no-link] [--rebuild] [--port N]   train here, no pod
run.py gpus   [--limit N]                          what exists, and what it costs
run.py up     [--gpu A,B] [--cloud SECURE|COMMUNITY]        create the pod only
run.py start  [--gpu A,B] [--cloud ...] [--no-link]
              [--rebuild] [--on-finish terminate|stop|keep] [--yes]
run.py ps                                          pods this tool created
run.py down   [--all] [--action terminate|stop] [--yes]
```

`--gpu` takes several, comma-separated: RunPod takes whichever is free, and
"no instances currently available" is the usual answer to a single choice.
`--rebuild` throws the cached environment away and builds it again.

---

## The idea

The dataset stays on your machine. The logs and the checkpoints come back to
your machine. Only the training happens elsewhere.

That is one class on each side:

| | on your machine | on the training machine |
|---|---|---|
| **data** | `app.dataset("train", ...)` | `DatasetWrapper("train")` |
| **output** | TensorBoard, checkpoints, artifact handlers | `LoggerWrapper` |

and, underneath both, one `Link` you can use directly for anything this
library does not model.

### `remote/train.py`

```python
from any_nn_runpod import DatasetWrapper, RunpodTrainer, session

class MyTrainer(RunpodTrainer):
    def train_step(self, step, batch, device, weight_dtype):
        images, labels = batch
        logits = self.models[0](images.to(device))
        loss = F.cross_entropy(logits, labels.to(device))
        return loss, {"accuracy": (logits.argmax(1) == labels.to(device)).float().mean()}

trainer = MyTrainer(output_dir=session.output_dir)
trainer.models = [model]
trainer.optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
trainer.train_dataloader = DatasetWrapper("train", precache=48, prepare=8)
trainer.batch_size = 64
trainer.epochs = 10

session.bind(trainer)   # link, logger, datasets -- one call
trainer.init()
trainer.train()
```

### `local/local.py`

```python
from any_nn_runpod import Local

app = Local(output_dir="out")

app.dataset("train", lambda: DataLoader(ds, batch_size=64, num_workers=8))

@app.on("samples")
def samples(payload, ctx):          # every handler is (payload, ctx)
    ctx.log_image("samples", grid(payload), ctx.step)
```

`local/` is never uploaded. `remote/` is uploaded whole -- put weights in it if
you want them there. "Whole" means whole: the only things left behind are
`__pycache__`, `.git`, virtualenvs and tool caches, all of them machine-made
and regenerable. A directory you named yourself always travels, whatever it is
called.

A run writes its output **beside** `remote/`, never inside it. That is what
lets the upload have no exceptions -- and it stops a checkpoint from being
deleted as stale by the next sync, which is what would happen to anything
written into a directory kept in step with your copy.

Reach anything you shipped with `session.path()`, which means the same thing
here and on the pod:

```python
model = Transformer.from_pretrained(session.path("weights", "base"))
```

A 8 GB directory of weights in `remote/` is uploaded once. Later runs hash it,
see it has not changed, and send nothing.

---

## precache

```python
DatasetWrapper("train", precache=48, prepare=8)
```

`precache` is a promise: **48 batches are kept ready on the training machine at
all times**. As the loop consumes them your machine tops the window back up.
It is a credit window, so it bounds both directions -- the GPU never waits for
the network, and a fast local machine cannot buffer an epoch into the pod's RAM.

`prepare` is how many batches go in one message. Bigger amortizes per-message
overhead; it must not exceed `precache`.

The unit is **one collated batch**, whatever its batch size. That is why
`batch_size > 1` is unremarkable here: the queue is counted in the same thing
the training loop consumes.

## Cutting the payload down

`pack` on your machine, `unpack` on the training machine. A pair.

```python
# local.py -- fp16 on the wire
app.dataset("train", factory, pack=lambda bs: [(x.half(), y) for x, y in bs])

# train.py -- fp32 again, on the GPU
DatasetWrapper("train", unpack=lambda bs: [(x.float(), y) for x, y in bs])
```

`unpack` runs on the training machine with the GPU available, so it is also the
place for anything expensive you would rather not send. (It replaces
`any_nn`'s `precache_dataset`, which is gone.)

## Sending work home

Some work should not happen on a rented GPU -- drawing a validation grid needs
your fonts, your class names, your dataset. Send the tensors instead:

```python
# train.py
self.send_artifact("samples", {"images": x, "predicted": p}, step)

# local.py
@app.on("samples")
def samples(payload, ctx):
    ctx.log_image("samples", grid(payload), ctx.step)
```

And in the other direction, when the training side needs something only your
machine has:

```python
# train.py
embeds = session.call("encode", {"prompts": [...]})

# local.py
@app.on("encode")
def encode(payload, ctx):
    return {"embeds": text_encoder(payload["prompts"])}
```

Same decorator, same signature. A handler that returns a value answers a
`call`; one that does not just handles an artifact.

**Only plain data crosses**: dict, list, tuple, str, bytes, numbers, None and
tensors. Your own classes will not unpickle on the other side -- it has never
imported them.

## Control

Lock files, wherever the output is (so, your machine when there is a link):

```bash
touch out/pause.lock            # pause; delete it to resume
touch out/do_eval.lock          # evaluate once
touch out/save_checkpoint.lock  # checkpoint once
touch out/stop.lock             # wind up cleanly
```

## Running with no local side at all

```bash
python run.py local --no-link
```

`DatasetWrapper` falls back to building the data itself, `LoggerWrapper` writes
next to `remote/`, artifacts are dropped with a note. The training script is
unchanged. Give the wrapper a `fallback=` and this mode works:

```python
DatasetWrapper("train", precache=48, fallback=lambda: DataLoader(...))
```

On a pod, a run with no local side keeps everything it produced **on the pod** --
that is the whole difference. So `run.py start --no-link` will not terminate it
afterwards however `on_finish` is set: it stops the pod instead, which releases
the GPU and keeps `/workspace`. Collect what you want, then `run.py down`.
Passing `--on-finish terminate` explicitly still terminates, on the grounds
that you said so.

## Configuration

`remote/anr.toml` -- the recipe. Travels with the code, so the same file builds
the venv here and on the pod.

```toml
[run]
entry = "train.py"
output_dir = "out"          # only used when there is no local side

[env]
python = "3.11"             # omit to accept whatever is already here
torch = "2.10.0"
torch_index = "https://download.pytorch.org/whl/cu128"
requirements = ["diffusers==0.37.1", "transformers==5.2.0"]

[pod]
image = "runpod/pytorch:1.1.0-cu1290-torch291-ubuntu2204"
gpu = ["NVIDIA GeForce RTX 4090", "NVIDIA RTX A5000"]
# "availability": RunPod picks whichever it has most of -- which can cost
# several times the one you listed first. "custom": the order above is a
# preference, so put the cheapest acceptable card at the top.
gpu_priority = "availability"
container_disk_gb = 60
```

Every `[pod]` field: `image`, `gpu`, `gpu_priority`, `gpu_count`,
`container_disk_gb`, `volume_gb`, `network_volume_id`, `cloud_type`
(`SECURE`/`COMMUNITY`), `data_centers`, `env`. `[env]` also takes
`torchvision` and `torchaudio`.

`anr.toml` in the project root -- paths and pod policy. Never uploaded.

```toml
[paths]
local = "local"             # local/local.py lives here
remote = "remote"           # uploaded whole
output = "local/out"        # where a linked run's results land

[pod]
name = "anr"                # pods are found and reused by this name
on_finish = "terminate"     # terminate | stop | keep
max_hours = 12              # 0 to disable
library_source = "git+https://github.com/YOU/any_nn_runpod.git"
```

### Where output goes

| run | output lands in |
|---|---|
| `run.py local` | `local/out/` (`[paths] output`) |
| `run.py local --no-link` | `out/`, beside `remote/` |
| `run.py start` | `local/out/` on **your** machine |
| `run.py start --no-link` | `/workspace/out` on the pod, beside `remote/` |

Never inside `remote/`: that directory is kept in step with your copy, so
anything written into it would be deleted as stale by the next sync.

## Pods and money

Every pod this tool creates carries `ANR_MANAGED=1`. Nothing here will list,
stop or terminate a pod without that marker, so **pods you made yourself are
invisible to it** — including one that happens to share the project's name.
`run.py ps` says how many others exist without showing them.

`RUNPOD_API_KEY` stays on your machine and is never put in a pod's environment:
a pod is rented from strangers and runs code pulled from the internet, and a
key that can create and delete pods is not something to leave on one. The
consequence is that pods are ended from here, by three things:

* the run reports finished or failed → `on_finish` decides what that means;
* Ctrl-C → it asks, unless `--yes`;
* `max_hours`, or the link going quiet with no bytes moving either way.

The honest gap: if **your** machine dies, the pod keeps running. So:

```bash
python run.py ps          # what is up, and what it costs
python run.py down --all  # end all of them -- still only the managed ones
```

A run with no local side is never terminated automatically, whatever
`on_finish` says: its output exists only on that pod, so it is stopped instead.

## Reproducibility

`trainer.seed` is applied by `init()`, which is after your model exists. Seed
before building it:

```python
from any_nn_runpod import seed_everything
seed_everything(0)          # first line of main()
```

Two runs on one machine are then bit-identical -- same scalars, same weights.
Across machines they are not, and cannot be: a different GPU and a different
torch build reduce in a different order. Measured on this example, same seed,
RTX 5090 vs A40: identical at step 0, ~1e-2 apart on loss after 1248 steps.

## Development

```bash
pip install -e .
pytest              # 100 tests, ~30s, no GPU and no RunPod account needed
```

The tests cover the whole pipeline except the RunPod REST calls themselves:
the supervisor runs over loopback exactly as it does on a pod, and the rule
that only pods this tool created are ever touched is checked against pod
records rather than against an account.

Verified against real pods: a training run end to end with batches streamed
from the local machine, and incremental sync across five pods at once.
