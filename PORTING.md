# Porting Text-to-LoRA to Blackwell (sm_120) + modern stack

Fork of [SakanaAI/text-to-lora](https://github.com/SakanaAI/text-to-lora) (upstream `8ba7749`,
June 2025), ported to run on an **NVIDIA RTX PRO 6000 Blackwell Workstation Edition**
(96GB, compute capability 12.0).

Ported 2026-09-17. Upstream is unchanged and tracked as the `upstream` remote, so
`git diff upstream/main` shows exactly what was altered.

## Why a port was necessary

Upstream pins `torch==2.4.0`, `vllm==0.5.4`, `xformers==0.0.27`, `transformers==4.46.2`,
Python 3.10, and a hardcoded `flash-attn` cu123/torch2.3 wheel. None of these ship
kernels for sm_120, so `uv sync` produces an environment that cannot address the GPU
at all.

## Working stack

| package | upstream | here |
|---|---|---|
| Python | 3.10 | **3.12** |
| torch | 2.4.0 | **2.11.0+cu130** |
| vllm | 0.5.4 | **0.25.1** |
| transformers | 4.46.2 | **5.17.0** |
| peft | — | 0.21.0 |

Install by letting vLLM pull its own torch — no PyTorch index URL needed:

```bash
uv venv --python 3.12 --seed
uv sync
uv pip install --no-deps src/fishfarm
uv pip install colorlog ninja
```

`ninja` must be on `PATH` (not merely in the venv): FlashInfer JIT-compiles its sampling
kernel at runtime via `subprocess`, which searches `PATH`. Activating the venv is enough;
calling `.venv/bin/python` directly is not.

## Changes

### 1. `pyproject.toml`, `.python-version`, `uv.lock`
`requires-python >= 3.12`; `vllm==0.25.1`; `transformers` unpinned. Stale lock deleted
and regenerated.

### 2. `src/fishfarm/fishfarm/models/vllm_model.py` — vLLM API
vLLM removed the `prompt_token_ids=` kwarg from `LLM.generate`. Token inputs now go
through `TokensPrompt` in the positional `prompts` argument.

Notably, **nothing else in the vLLM surface needed changing** across 20 releases:
`prompts` and `lora_request` kept their names, and `LoRARequest(name, id, path)` kept
its positional order, so `vllm_eval.py` needed no API changes.

### 3. `src/hyper_llm_modulator/utils/model_loading.py` — attention backend
`attn_implementation` now tries `flash_attention_2` and falls back to `sdpa` when
flash-attn is not importable. Upstream flash-attention does not support sm_120
(Dao-AILab issues #1987, #2307).

This is inert for correctness: `generate_lora.py` never calls the base model's forward
(it loads it only to count layers), and evaluation runs through vLLM, which uses its own
attention backend — vLLM does run FlashAttention-2 on this card via its bundled
`vllm-flash-attn`. FA2 would only matter for training throughput.

### 4. `src/hyper_llm_modulator/utils/model_loading.py` — `get_extended_attention_mask`
transformers 5 removed `ModuleUtilsMixin.get_extended_attention_mask`, which
gte-large-en-v1.5's `trust_remote_code` code still calls. Reattached the 4.x
implementation (non-decoder branch).

### 5. `src/hyper_llm_modulator/utils/model_loading.py` — uninitialized buffers ⚠️
**The most important fix in this port.** See "The silent bug" below.

### 6. `src/hyper_llm_modulator/utils/model_loading.py` — tokenizer `trust_remote_code`
The embedding *model* load passed `trust_remote_code=True` but the *tokenizer* load did
not, causing an interactive `y/N` prompt that hangs unattended runs.

### 7. `src/hyper_llm_modulator/vllm_eval.py` — dataset id
`eval_gsm8k` hardcoded the bare id `"gsm8k"`; the Hub now requires `namespace/name`.
Switched to the repo's own `DS_PATHS`/`DS_KWARGS` tables, which already mapped it
correctly to `openai/gsm8k`.

### 8. `src/hyper_llm_modulator/vllm_eval.py` — script-based datasets
`datasets` 4+ removed support for loading datasets via Python scripts
(`RuntimeError: Dataset scripts are no longer supported, but found piqa.py`). `ybisk/piqa`
still ships `piqa.py`, so `DS_KWARGS["piqa"]` now reads the Hub's auto-converted parquet
branch via `revision="refs/convert/parquet"`. `allenai/winogrande` is already parquet and
needed no change.

### 9. `src/hyper_llm_modulator/hyper_modulator.py` — rsLoRA scaling ⚠️
**The bug that broke the reproduction.** See "The scaling bug" below.

### 10. `pyproject.toml` — Gradio major version
The web UI declares `gradio>=5.29.0`, which now resolves to Gradio 6. Gradio 6 removed
`Chatbot(type=...)` and moved `theme` from the `Blocks` constructor to `launch()`, so
`webui/app.py` fails immediately with
`TypeError: Chatbot.__init__() got an unexpected keyword argument 'type'`.
Pinned to `>=5.29.0,<6`: gradio is used only by `webui/` (53 `gr.*` call sites) and
nothing in the training or evaluation path touches it.

Note the UI generates adapters through `gen_and_save_lora` -> `save_lora`, so it picks up
the rsLoRA fix in change 9 automatically and does not emit overdriven adapters.

## Running fully offline

`scripts/prefetch_offline.py` caches everything the repo touches; after that,
`export HF_HUB_OFFLINE=1` makes the whole stack run with no network. Verified end to
end: all 10 benchmarks and all 510 Lots-of-LoRAs task datasets load from cache.

```bash
python scripts/prefetch_offline.py --bases all   # ~120GB, models dominate
export HF_HUB_OFFLINE=1
python scripts/prefetch_offline.py --verify      # exits 0 when genuinely offline-ready
```

| component | size |
|---|---|
| 3 base models (Mistral-7B, Llama-3.1-8B, Gemma-2-2B) + gte encoder | ~118 GB |
| 510 Lots-of-LoRAs task datasets | 2.8 GB |
| 10 benchmark datasets (eval + train splits) | ~100 MB |
| evalplus HumanEval+/MBPP+ | 1.0 GB |
| 4 T2L checkpoints | ~1.9 GB |

Two things that make this non-trivial, both learned the hard way:

- **Datasets must be fetched through `load_dataset` with the exact kwargs the repo uses**
  (read from `tasks/*/metadata.yaml`). The datasets cache is keyed by
  `(path, name, split)`, so a bare repo download still leaves `load_dataset` reaching for
  the network at run time.
- **`mbpp` has configs `['full','sanitized']` and no default**, so an unnamed config is
  ambiguous offline and raises `ValueError` even when cached. Both are pinned explicitly.
  (The repo actually reads MBPP through evalplus, not HF, but caching both costs nothing.)

Keep `--workers` low: 8 parallel dataset pulls tripped Hub rate limiting at ~330/510.
3 workers with retries completes all 510 cleanly.

## The scaling bug (the one that broke reproduction)

T2L's generated adapters carry `use_rslora: true`, but the weights are calibrated for
**plain LoRA scaling, alpha/r**. vLLM 0.5.4 -- the version upstream used -- had no rsLoRA
handling and applied `alpha/r` unconditionally. Modern vLLM (`vllm/lora/peft_helper.py`)
and modern peft both *honour* the flag and apply `alpha/sqrt(r)`, overdriving every
generated adapter by sqrt(r) -- **2.83x at r=8** (5.657 instead of 2.0).

A newly added upstream feature broke the reproduction. The config never changed; the
interpretation of it did.

**Evidence.** GSM8K with the adapter unchanged and only `lora_alpha` varied:

| effective scaling | GSM8K |
|---|---|
| 0.707 | 44.05 |
| 1.414 | 45.11 |
| 2.828 | **45.79** |
| 4.243 | 39.42 |
| 5.657 (rsLoRA -- what we were applying) | 27.82 |

The peak brackets `alpha/r = 2.0`. Paper reports 44.02, base model 41.02.

**Symptom.** At 2.83x the adapter perturbs the base weights enormously -- ‖dW‖/‖W‖ of
0.32-0.93 (layer 31 q_proj: 0.93), where a normally-trained LoRA sits at 0.01-0.05.
Generation becomes *terse*, not truncated: the model compresses chain-of-thought
(36-108 tokens vs the base model's 152-270) and drops the intermediate arithmetic that
CoT accuracy depends on -- writing "97 eggs" where 16-3-4=9. Multiple-choice tasks
survive because they only need one token to be right; free-form generation collapses.

**Fix.** `save_lora()` writes `use_rslora=False` on generated adapters, matching how the
weights were actually calibrated. Existing adapters can be corrected in place by editing
the flag alone -- the weights are unchanged, only the interpretation.

**Generalise this.** When reproducing older LoRA work on a modern stack, verify the
*effective scaling factor*, not merely that the adapter loads. A silent 2.83x error
looks like a mediocre result rather than a bug. Measuring ‖dW‖/‖W‖ against the 1-5%
norm is a fast sanity check.

## The silent bug (read this one)

Under transformers 5, `Alibaba-NLP/gte-large-en-v1.5` — the encoder T2L conditions on —
**silently produces garbage task embeddings**.

transformers 5 materializes models from the meta device and only fills buffers present
in the checkpoint. gte builds three **non-persistent** buffers in `__init__`:

| buffer | expected | actual under transformers 5 |
|---|---|---|
| `embeddings.position_ids` | `arange(8192)` | uninitialized (±9.2e18) |
| `rotary_emb.inv_freq` | RoPE frequencies | uninitialized (max 7.5e28) |
| `rotary_emb.cos_cached` / `sin_cached` | RoPE tables | **all zeros** |

None appear in the checkpoint, so none are restored. Zeroed cos/sin caches remove all
positional information.

**Why it is dangerous:** no crash on GPU. Every sentence collapses to nearly the same
vector — unrelated sentences at cosine 1.000, *higher* than true paraphrases, with ~94%
of each embedding being a shared mean component. The hypernetwork then emits a
well-formed, correctly-shaped, coherent-but-unspecialized adapter. GSM8K scored **18.5%**
against a paper value of 44.02 and a base model of 41.02 — worse than no adapter at all.
On CPU it does raise `IndexError: index 123653633196144 is out of bounds`, which is what
exposed it.

`_fix_uninitialized_buffers()` rebuilds all three using the modules' own initialisation
logic. After the fix, paraphrase similarity 0.757 vs unrelated 0.387/0.438.

**Generalise this.** Any model loaded via `trust_remote_code` under transformers 5 is
suspect. Diagnose by scanning buffers before trusting any result:

```python
for name, buf in model.named_buffers():
    if not torch.isfinite(buf).all() or buf.abs().max() > 1e6 or buf.abs().max() == 0:
        print("SUSPECT:", name)
```

## Reproduction status

**Fully reproduced.** All 10 benchmarks match upstream to a mean absolute error of
**0.18 points** (worst case 0.69), once the rsLoRA scaling is corrected.

Base model without any adapter, confirming the harness independently:

| | GSM8K |
|---|---|
| base Mistral-7B-Instruct-v0.2 | **40.94** |
| upstream README | 41.02 |

T2L-generated adapters, `eval_descs` group, mean of 3 adapters per task. "overdriven" is
what the same adapters score when `use_rslora=true` is honoured (2.83x too strong):

| task | overdriven | **corrected** | upstream | delta |
|---|---|---|---|---|
| BoolQ | 83.38 | **84.63** | 84.62 | +0.01 |
| HellaSwag | 61.33 | **67.09** | 67.08 | +0.01 |
| ARC-c | 74.69 | **77.39** | 77.42 | −0.03 |
| ARC-e | 86.77 | **89.16** | 89.20 | −0.04 |
| WinoGrande | 61.67 | **63.19** | 63.14 | +0.05 |
| PIQA | 79.47 | **82.21** | 82.32 | −0.11 |
| OpenBookQA | 72.33 | **74.87** | 75.07 | −0.20 |
| MBPP | 38.35 | **48.96** | 48.71 | +0.25 |
| HumanEval | 18.70 | **38.21** | 38.62 | −0.41 |
| GSM8K | 26.28 | **44.71** | 44.02 | +0.69 |

A single scaling flag accounted for every discrepancy. Before the fix the failures looked
like two separate problems — free-form generative tasks collapsing below the un-adapted
base model, and multiple-choice tasks sitting 1–6 points low. Both were the same 2.83x
overdrive. Multiple-choice tasks degraded gracefully because they only need one token to
be right; generation collapsed because sustained output compounds the distortion.

## Reproducibility caveats

Upstream notes that vLLM's LoRA application is non-deterministic even with a fixed seed,
and that they re-trained all baselines "due to a small mismatch between the specific
package version combinations". This port is 20 vLLM releases, one transformers major
version and two torch major versions ahead of upstream, so the residual 0.18-point mean
error is well within that noise.

**A correction, recorded deliberately.** An earlier revision of this document diagnosed
the generative failure as "premature termination" -- the model stopping before finishing.
That was wrong. Full generations show complete answers that are merely *terse*; the model
skips the intermediate arithmetic rather than being cut off. The misdiagnosis came from
reading truncated log excerpts instead of whole outputs. The `use_rslora` hypothesis was
also checked early and wrongly dismissed, because modern vLLM *does* handle the flag
correctly -- which is precisely what breaks compatibility with results produced by a
version that ignored it.
