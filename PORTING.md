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

Base model reproduces essentially exactly, which validates the whole harness — dataset,
chat template, prompting, sampling, answer extraction and vLLM:

| | GSM8K |
|---|---|
| base Mistral-7B-Instruct-v0.2, no adapter | **40.94** |
| upstream README | 41.02 |

With the T2L-generated adapter, results split sharply by required output length
(`eval_descs` group, `mistral_7b_t2l`):

| task | output required | ours | README T2L | base |
|---|---|---|---|---|
| BoolQ | yes/no | 83.38 | 84.62 | 71.56 |
| ARC-e | one letter | 86.77 | 89.20 | 77.74 |
| ARC-c | one letter | 74.69 | 77.42 | 65.79 |
| HellaSwag | pick of 4 | 61.33 | 67.08 | 49.64 |
| GSM8K | multi-step CoT | **26.28** | 44.02 | 41.02 |
| HumanEval | a function | **18.70** | 38.62 | 39.02 |

**Short-output tasks reproduce** (BoolQ within 1.2 points, large gains over base).
**Long-output tasks fail**, falling below the un-adapted base model.

The failure mode is **premature termination**, not truncation — outputs average ~250
characters against a 512-token cap. The model produces correct intermediate reasoning
and stops before the final step:

```
"She eats three eggs for breakfast / She bakes four muffins
 / She sells the remainder of 9 eggs"          -> predicted 9, answer 18
```

`extract_answer_number` then picks up the last number seen, an intermediate value.

The degradation scales monotonically with required output length (BoolQ −1.2, ARC-e
−2.4, ARC-c −2.7, HellaSwag −5.8, GSM8K −17.7, HumanEval −19.9), which is one coherent
signature rather than a diffuse shortfall.

Evidence that the encoder and hypernetwork are otherwise sound: descriptions borrowed
from the *wrong* task (`other_train_descs`) degrade results dramatically (GSM8K 14.13,
HumanEval 4.07), so the task embedding genuinely carries signal and the hypernetwork
genuinely specialises.

**Open question:** why the adapter biases toward early EOS in free-form generation. The
base model generates complete answers through the identical harness, so this is
adapter-induced rather than an eval-plumbing problem.

## Reproducibility caveats

Upstream notes that vLLM's LoRA application is non-deterministic even with a fixed seed,
and that they re-trained all baselines "due to a small mismatch between the specific
package version combinations". This port is 20 vLLM releases, one transformers major
version and two torch major versions ahead of upstream, so small drift on the
short-output tasks is expected. The long-output failure is far too large to be explained
that way.
