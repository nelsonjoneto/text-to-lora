import logging
from math import sqrt
import os

import torch
from peft import PeftModel
from peft import get_peft_config as _get_peft_config
from peft.utils import PeftType
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

from hyper_llm_modulator.utils.pooling import get_pooling_fn
from hyper_llm_modulator.utils.preprocessing import add_full_stop, apply_sfr_template

logger = logging.getLogger()


def get_model_and_tokenizer(
    model_path,
    train,
    requires_grad,
    use_flash_attn=True,
    peft_config=None,
    model_kwargs=None,
    tokenizer_kwargs=None,
    device="cuda:0",
    dtype=torch.bfloat16,
):
    model = get_model(
        model_path,
        train,
        requires_grad,
        use_flash_attn,
        peft_config,
        model_kwargs,
        device,
        dtype,
    )
    tokenizer = get_tokenizer(model_path, tokenizer_kwargs, peft_config, train)
    return model, tokenizer


def get_tokenizer(model_path, tokenizer_kwargs=None, peft_config=None, train=False):
    # NOTE: lora models don't have tokenizer config in the folder

    padding_side = "left" if not train else "right"
    if peft_config:
        model_path = peft_config.base_model_name_or_path

    if tokenizer_kwargs is None:
        tokenizer_kwargs = {}

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, padding_side=padding_side, **tokenizer_kwargs
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    template_path = f"chat_templates/{model_path}/chat_template.jinja"
    assert os.path.exists(template_path), (
        f"Chat template not found for {model_path}.\n"
        "We assume a specfic form of chat template for consistency between models. "
        "Please use the templates provided."
    )
    print(f"Loading chat template from {template_path}")
    chat_template = open(template_path).read()
    chat_template = chat_template.replace("    ", "").replace("\n", "")
    tokenizer.chat_template = chat_template

    tokenizer.add_eos_token = False
    tokenizer.truncation_side = "left"
    return tokenizer


def get_model(
    model_path,
    train,
    requires_grad,
    use_flash_attn=True,
    peft_config=None,
    model_kwargs=None,
    device="cuda:0",
    dtype=torch.bfloat16,
):
    model_init_kwargs = dict(
        pretrained_model_name_or_path=model_path,
        device_map=device,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    if model_kwargs is not None:
        model_init_kwargs.update(model_kwargs)
    if use_flash_attn:
        try:
            import flash_attn  # noqa: F401

            model_init_kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            # No flash-attn wheel exists for torch 2.11/cu130 on sm_120 (Blackwell).
            # SDPA dispatches to fused kernels there and is equivalent for our use:
            # this model is only used for LoRA generation/training, while vLLM uses
            # its own attention backend during evaluation.
            model_init_kwargs["attn_implementation"] = "sdpa"
    if train:
        # for training disable cache
        model_init_kwargs["use_cache"] = False
    logger.debug(f"Model init kwargs: {model_init_kwargs}")
    model = AutoModelForCausalLM.from_pretrained(**model_init_kwargs)
    if peft_config is not None:
        model = PeftModel(model, peft_config)
    model.train(train)
    for param in model.parameters():
        param.requires_grad = requires_grad
    return model


def get_peft_config(model_dir, peft_type, **kwargs):
    peft_type = peft_type.upper()
    assert peft_type in [PeftType.LORA, PeftType.VERA]

    peft_conf_kwargs = dict(
        r=8 if peft_type == PeftType.LORA else 64,
        peft_type=peft_type,
        base_model_name_or_path=model_dir,
        task_type="CAUSAL_LM",
    )

    peft_conf_kwargs[f"{peft_type.lower()}_dropout"] = 0.05

    if peft_type == PeftType.LORA:
        peft_conf_kwargs["use_rslora"] = True
        peft_conf_kwargs["lora_alpha"] = peft_conf_kwargs["r"] * 2

    peft_conf_kwargs.update(kwargs)
    peft_config = _get_peft_config(peft_conf_kwargs)
    return peft_config


def _fix_uninitialized_buffers(model):
    """transformers 5 materializes models from the meta device and only fills buffers
    that appear in the checkpoint. gte builds `position_ids`, `inv_freq` and the RoPE
    `cos_cached`/`sin_cached` in __init__ as NON-PERSISTENT buffers, so under
    transformers 5 they come back as uninitialized memory (position_ids, inv_freq) or
    as zeros (the cos/sin caches). Zeroed RoPE caches strip out all positional
    information, which collapses every input to nearly the same embedding.
    Rebuild them with the modules' own initialisation logic."""
    emb = getattr(model, "embeddings", None)
    if emb is None:
        return model

    pos = getattr(emb, "position_ids", None)
    if pos is not None:
        expected = torch.arange(pos.numel(), device=pos.device, dtype=pos.dtype)
        if not torch.equal(pos, expected):
            logger.warning("rebuilding uninitialized gte position_ids buffer")
            emb.register_buffer("position_ids", expected, persistent=False)

    rot = getattr(emb, "rotary_emb", None)
    if rot is not None and getattr(rot, "inv_freq", None) is not None:
        cos = getattr(rot, "cos_cached", None)
        broken = (
            not torch.isfinite(rot.inv_freq).all()
            or float(rot.inv_freq.abs().max()) > 1e6
            or (cos is not None and float(cos.abs().max()) == 0.0)
        )
        if broken:
            logger.warning("rebuilding uninitialized gte RoPE buffers (inv_freq, cos/sin caches)")
            dev = rot.inv_freq.device
            # mirrors RotaryEmbedding.__init__
            rot.register_buffer(
                "inv_freq",
                1.0 / (rot.base ** (torch.arange(0, rot.dim, 2).float().to(dev) / rot.dim)),
                persistent=False,
            )
            seq_len = rot.max_position_embeddings
            if getattr(rot, "scaling_factor", None):
                # NTKScalingRotaryEmbedding.__init__ re-caches at max_pos * scaling_factor
                seq_len = int(seq_len * rot.scaling_factor)
            rot._set_cos_sin_cache(seq_len, dev, torch.get_default_dtype())
    return model


def _restore_get_extended_attention_mask(model):
    """transformers 5.x removed ModuleUtilsMixin.get_extended_attention_mask, but
    gte-large-en-v1.5's `trust_remote_code` modeling.py still calls it. Reattach the
    4.x implementation (non-decoder branch) so the remote code keeps working."""
    import types

    if hasattr(model, "get_extended_attention_mask"):
        return model

    def get_extended_attention_mask(self, attention_mask, input_shape=None, device=None, dtype=None):
        if dtype is None:
            dtype = self.dtype
        if attention_mask.dim() == 3:
            extended = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended = attention_mask[:, None, None, :]
        else:
            raise ValueError(f"Wrong shape for attention_mask (shape {attention_mask.shape})")
        extended = extended.to(dtype=dtype)
        return (1.0 - extended) * torch.finfo(dtype).min

    model.get_extended_attention_mask = types.MethodType(get_extended_attention_mask, model)
    return model


def get_emb_model_and_fns(emb_model_name, device):
    emb_model = AutoModel.from_pretrained(
        emb_model_name,
        device_map=device,
        torch_dtype=torch.float32 if "gte" in emb_model_name else torch.bfloat16,
        trust_remote_code=True,
    ).eval()
    _restore_get_extended_attention_mask(emb_model)
    _fix_uninitialized_buffers(emb_model)
    # the model load above already passes trust_remote_code; without it here the
    # tokenizer blocks on an interactive y/N prompt and hangs unattended runs
    emb_tokenizer = AutoTokenizer.from_pretrained(emb_model_name, trust_remote_code=True)
    if emb_tokenizer.pad_token_id is None:
        emb_tokenizer.pad_token_id = emb_tokenizer.eos_token_id
    task_desc_format_fn = add_full_stop
    if "SFR" in emb_model_name:
        task_desc_format_fn = apply_sfr_template
        pooling_fn = get_pooling_fn("last_token")
    elif "gte" in emb_model_name:
        pooling_fn = get_pooling_fn("cls")
    return emb_model, emb_tokenizer, task_desc_format_fn, pooling_fn
