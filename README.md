![Project Poster](Poster.png)

# Domain Expert Formalizers

See `Domain_Expert_Formalizers.pdf` for results.

Datasets produced are available on [Hugging Face](https://huggingface.co/datasets/connorolson/lean4-subgoal-completions), which is also where you can find the adapters for the [generalist](https://huggingface.co/connorolson/qwen35-9b-lean4-generalist-lora), [geometry](https://huggingface.co/connorolson/qwen35-9b-lean4-geometry-lora), and [nongeometry](https://huggingface.co/connorolson/qwen35-9b-lean4-nongeometry-lora) models respectively.

## Reproducing Results

To reproduce all my results end to end, you can use `./scripts/process/*` to generate the synthetic data, and train models using the scripts `./scripts/*` with the Lean harness provided in `src/lean_eval/`, and the Lean environment in `./lean/`.

For a mere evaluation of the models, all the prompts used for training and testing are described in `Domain_Expert_Formalizers.pdf`. Any Lean harness will do, but `lean/` and `src/lean_eval/` should contain everything you need.

### Python

Use Python 3.12, and use [`uv`](https://github.com/astral-sh/uv) instead of `pip` with a virtual environment:

```bash
# install `uv`
curl -LsSf https://astral.sh/uv/install.sh | sh

# create virtual environment
uv venv --python 3.12

# active virtual environment (Linux)
source .venv/bin/activate

# install elan
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
source $HOME/.elan/env

# install dependencies
uv pip install -r requirements.txt

# install the local package so scripts can import `lean_eval`
uv pip install -e .
```

### LeanDojo

This project uses `lean-dojo-v2` since at time of work `lean-dojo` is deprecated. However, `lean-dojo-v2` does some unfriendly things which are adjusted in our runtime environment.

You will need to set some environment variables (see `.env.sample`) to get LeanDojo to store files in a persistent location with enough memory for this project, at which point you can run `trace_mathlib.py`. It will fail after producing a malversioned `.../ExtractData.lean` file, which you can replace with `./scripts/ExtractData.lean`. After deleting the stale binary `rm ~/.cache/lean_dojo/.../ExtractData.olean`, you can run extraction on Mathlib manually via `lake env lean --run ExtractData.lean`, and then use `./scripts/force_parse.py` to generate XML files. If you have an empty `raid/data` directory, delete it, then you can use `trace_mathlib.py` to generate the traces used in this project for synthetic data generation.
