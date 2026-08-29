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
you want them there.

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
gpu = ["NVIDIA GeForce RTX 4090"]
container_disk_gb = 60
```

`anr.toml` in the project root -- paths and pod policy. Never uploaded.

```toml
[paths]
local = "local"
remote = "remote"
output = "local/out"

[pod]
name = "anr"
on_finish = "terminate"     # terminate | stop | keep
max_hours = 12
```

`.env` holds `RUNPOD_API_KEY`. **It stays on your machine and is never put in a
pod's environment** -- a pod is rented from strangers, and a key that can create
and delete pods is not something to leave on one. Pods are therefore ended from
here.

## What happened to easy_nn

`easy_nn` shipped your trainer to the pod as pickled bytecode, which bought
"the executor runs any script" at a steep price: Python and torch had to match
to the minor version on both sides, the pod rebuilt its own interpreter to make
that true, and the model was uploaded again on every run.

Here `remote/` is deployed as files and imported normally. No bytecode crosses,
so nothing has to match; the recipe says what to install and that is that. What
survived is the part that was actually valuable: the tensor-aware codec, the
framed protocol, and credit-based flow control.

## Development

```bash
pip install -e .
pytest              # 93 tests, ~30s, no GPU and no RunPod account needed
```

The tests cover the whole pipeline except the RunPod REST calls themselves:
the supervisor runs over loopback exactly as it does on a pod, and the rule
that only pods this tool created are ever touched is checked against pod
records rather than against an account.
