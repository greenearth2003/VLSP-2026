"""Speaker embedding model zoo.

Every model returned by `load_model` takes a float waveform batch [B, T] at
16 kHz in [-1, 1] and returns embeddings [B, D] (not L2-normalised).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio

ROOT = Path(__file__).resolve().parent.parent
PRETRAINED = ROOT / "pretrained"
REDIMNET_V1_SRC = ROOT / ".cache/torch/hub/IDRnD_ReDimNet_master"
REDIMNET_V2_SRC = ROOT / ".cache/torch/hub/PalabraAI_redimnet2_main"
WESPEAKER_SRC = ROOT / "third_party/wespeaker"


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ----------------------------------------------------------------------------
# ReDimNet v1 (IDRnD)
# ----------------------------------------------------------------------------
def _load_redimnet_v1(ckpt: Path) -> nn.Module:
    sys.path.insert(0, str(REDIMNET_V1_SRC))
    from redimnet.model import ReDimNetWrap

    full = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = ReDimNetWrap(**full["model_config"])
    res = model.load_state_dict(full["state_dict"])
    assert not res.missing_keys and not res.unexpected_keys, res
    return model


# ----------------------------------------------------------------------------
# ReDimNet2 (PalabraAI) -- weights only available as MLX safetensors on HF,
# converted to PyTorch layout here (MLX conv = channels-last).
# ----------------------------------------------------------------------------
def build_redimnet2(config: dict) -> nn.Module:
    sys.path.insert(0, str(REDIMNET_V2_SRC))
    from redimnet2.redimnet2 import ReDimNet2Wrap

    return ReDimNet2Wrap(**config)


def _load_redimnet2_pt(ckpt: Path) -> nn.Module:
    """Official PalabraAI GitHub release checkpoint ({model_config, state_dict})."""
    full = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = build_redimnet2(full["model_config"])
    res = model.load_state_dict(full["state_dict"])
    assert not res.missing_keys and not res.unexpected_keys, res
    model.model_config = full["model_config"]
    return model


def _load_redimnet2_mlx(mlx_dir: Path) -> nn.Module:
    from safetensors.numpy import load_file

    config = json.loads((mlx_dir / "config.json").read_text())
    model = build_redimnet2(config)
    sd = model.state_dict()
    weights = load_file(str(mlx_dir / "weights.safetensors"))
    new_sd = {}
    for k, v in sd.items():
        if k not in weights:
            assert k.endswith("num_batches_tracked"), k
            new_sd[k] = v
            continue
        w = torch.from_numpy(weights[k].copy())
        if tuple(w.shape) != tuple(v.shape):
            if w.ndim == 4:  # Conv2d: (O, H, W, I) -> (O, I, H, W)
                w = w.permute(0, 3, 1, 2)
            elif w.ndim == 3:  # Conv1d: (O, K, I) -> (O, I, K)
                w = w.permute(0, 2, 1)
            assert tuple(w.shape) == tuple(v.shape), (k, w.shape, v.shape)
        new_sd[k] = w.contiguous().to(v.dtype)
    model.load_state_dict(new_sd)
    model.model_config = config
    return model


# ----------------------------------------------------------------------------
# WeSpeaker models (fbank front-end + CMN, wav scaled to int16 range)
# ----------------------------------------------------------------------------
class WeSpeakerWrap(nn.Module):
    def __init__(self, backbone: nn.Module, num_mel_bins: int = 80):
        super().__init__()
        self.backbone = backbone
        self.num_mel_bins = num_mel_bins

    def fbank(self, wav: torch.Tensor) -> torch.Tensor:
        feats = []
        for w in wav:
            f = torchaudio.compliance.kaldi.fbank(
                w.unsqueeze(0) * (1 << 15),
                num_mel_bins=self.num_mel_bins,
                frame_length=25,
                frame_shift=10,
                dither=0.0,
                sample_frequency=16000,
                window_type="hamming",
                use_energy=False,
            )
            feats.append(f - f.mean(dim=0, keepdim=True))
        return torch.stack(feats)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        out = self.backbone(self.fbank(wav))
        return out[-1] if isinstance(out, tuple) else out


def _load_wespeaker(model_dir: Path, ckpt_name: str) -> nn.Module:
    import yaml

    import types

    # Register bare packages so importing wespeaker.models.* skips the heavy
    # package __init__ (which pulls CLI deps such as silero_vad).
    for pkg, sub in (("wespeaker", "wespeaker"), ("wespeaker.models", "wespeaker/models")):
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = [str(WESPEAKER_SRC / sub)]
            sys.modules[pkg] = m
    cfg = yaml.safe_load((model_dir / "config.yaml").read_text())
    name = cfg["model"]
    if name.startswith("SimAM"):
        import wespeaker.models.samresnet as mod
    elif name.startswith("ECAPA"):
        import wespeaker.models.ecapa_tdnn as mod
    else:
        import wespeaker.models.resnet as mod
    backbone = getattr(mod, name)(**cfg["model_args"])
    sd = torch.load(model_dir / ckpt_name, map_location="cpu", weights_only=False)
    sd = {k: v for k, v in sd.items() if not k.startswith("projection")}
    res = backbone.load_state_dict(sd, strict=False)
    assert not res.missing_keys, res.missing_keys
    return WeSpeakerWrap(backbone)


# ----------------------------------------------------------------------------
REGISTRY = {
    # name: (loader, path)
    "redimnet_M_mix": ("v1", "redimnet_v1/M-vb2+vox2+cnc-ft_mix.pt"),
    "redimnet_S_mix": ("v1", "redimnet_v1/S-vb2+vox2+cnc-ft_mix.pt"),
    "redimnet_b3_vox2": ("v1", "redimnet_v1/b3-vox2-ft_lm.pt"),
    "redimnet_b4_vox2": ("v1", "redimnet_v1/b4-vox2-ft_lm.pt"),
    "redimnet_b5_vox2": ("v1", "redimnet_v1/b5-vox2-ft_lm.pt"),
    "redimnet2_b4_vox2": ("v2mlx", "causal__redimnet2-b4-vox2-lm-mlx"),
    "redimnet2_b3_vox2": ("v2mlx", "causal__redimnet2-b3-vox2-lm-mlx"),
    "redimnet2_b3_mix": ("v2mlx", "causal__redimnet2-b3-vb2-vox2-cnc2-lm-mlx"),
    # official PalabraAI GitHub release v1.0.0 weights
    "redimnet2_b4_vox2_pt": ("v2pt", "redimnet2/b4-vox2-lm.pt"),
    "redimnet2_b6_mix_teacher_pt": ("v2pt", "redimnet2/b6-vb2+vox2+cnc2_v0-lm.pt"),
    "wespeaker_r34_lm": ("wespeaker:avg_model", "Wespeaker__wespeaker-voxceleb-resnet34-LM"),
    "wespeaker_samr34_vb2": ("wespeaker:avg_model.pt", "gaunernst__wespeaker-voxblink2-samresnet34-ft"),
    "wespeaker_ecapa512_lm": ("wespeaker:avg_model.pt", "Wespeaker__wespeaker-ecapa-tdnn512-LM"),
}


C2_PKG = Path("/mnt/tsharp/thucth/dataset/vlsp2026-sred/incoming/c2pkg")  # teammate's C2 package (code + weights)


def _load_c2(pkg: Path) -> nn.Module:
    """Teammate's C2: ReDimNet2-B6 with 14 low-rank layers (6,917,480 params), built from its package and loaded
    weights-only (no arbitrary unpickling). Waveform [B, T] in, raw 192-d embedding out."""
    import sys
    for p in (str(pkg), str(pkg / "third_party")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from c2q3 import model as M
    from redimnet2.redimnet2 import ReDimNet2Wrap
    st = torch.load(pkg / "c2q3/assets/c2_model.pt", map_location="cpu", weights_only=True)
    model = ReDimNet2Wrap(**dict(st["b6_cfg"]))
    for name, r in st["ranks"].items():
        m = model.get_submodule(name)
        kind = "conv1d" if isinstance(m, nn.Conv1d) else "linear"
        M._set_module(model, name, M.LowRank(kind, m.in_channels if kind == "conv1d" else m.in_features,
                                             m.out_channels if kind == "conv1d" else m.out_features, r))
    for name in st["drop"]:
        M._set_module(model, name, nn.Identity())
    model.load_state_dict(st["model"])
    return model


def load_model(name: str) -> nn.Module:
    """Load a pretrained model by registry name, or a fine-tuned checkpoint path (.pt
    produced by sred.train)."""
    if name not in REGISTRY and Path(name).exists():
        return load_finetuned(Path(name))
    if name == "c2":
        return _load_c2(C2_PKG).eval()
    kind, rel = REGISTRY[name]
    path = PRETRAINED / rel
    if kind == "v1":
        model = _load_redimnet_v1(path)
    elif kind == "v2pt":
        model = _load_redimnet2_pt(path)
    elif kind == "v2mlx":
        model = _load_redimnet2_mlx(path)
    elif kind == "v2gguf":
        model = _load_redimnet2_gguf(path)
    elif kind.startswith("wespeaker:"):
        model = _load_wespeaker(path, kind.split(":", 1)[1])
    elif kind.startswith("3ds:"):
        model = _load_3dspeaker(path, kind.split(":", 1)[1])
    else:
        raise ValueError(kind)
    return model.eval()


def load_finetuned(ckpt: Path) -> nn.Module:
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if state.get("kind") == "fakequant_module":  # whole module saved by scripts/quant_eval.py (static INT8 acts)
        return state["module"].eval()
    model = load_model(state["base_model"])
    if state.get("prune"):  # structurally pruned checkpoint (sred/prune.py): rebuild the slim architecture first
        from .prune import apply_indices
        apply_indices(model, state["prune"])
    if state.get("expand"):  # function-preserving block expansion (sred/expand.py): rebuild the deeper architecture
        from .expand import expand_model
        expand_model(model, state["expand"], zero_init=False)
    if state.get("denoiser"):  # learned denoising front-end (sred/denoise.py)
        from .denoise import attach_denoiser
        attach_denoiser(model, state["denoiser"])
    model.load_state_dict(state["model"])
    return model.eval()


# ----------------------------------------------------------------------------
# ReDimNet2-B6 (vb2+vox2+cnc2_v0-lm) from the OpenASR GGUF pack (fp16 weights).
# Approved checkpoint, 12.3M params -> usable only as a TEACHER (> 7M).
# ----------------------------------------------------------------------------
def _load_redimnet2_gguf(path: Path) -> nn.Module:
    import inspect

    import numpy as np
    from gguf import GGUFReader

    r = GGUFReader(str(path))
    f = r.fields["redimnet2.model_config_json"]
    cfg = json.loads(bytes(f.parts[f.data[0]]).decode())
    # OpenASR flattens stages as [sf, st, n_blocks, conv_exp, att_red]; kernels are (3,3)
    cfg["stages_setup"] = [((int(a), int(b)), int(n), int(e) if float(e).is_integer() else e, [(3, 3)], int(red))
                           for a, b, n, e, red in cfg["stages_setup"]]
    sys.path.insert(0, str(REDIMNET_V2_SRC))
    from redimnet2.redimnet2 import ReDimNet2Wrap

    ok = set(inspect.signature(ReDimNet2Wrap.__init__).parameters)
    cfg = {k: v for k, v in cfg.items() if k in ok}
    cfg.setdefault("causal", "none")
    model = ReDimNet2Wrap(**cfg)
    tensors = {t.name: t for t in r.tensors}
    sd = model.state_dict()
    new_sd = {}
    for k, v in sd.items():
        if k not in tensors:
            # deterministic front-end buffers (STFT kernels, mel banks) are not packed
            assert k.endswith("num_batches_tracked") or k.startswith("spec."), k
            new_sd[k] = v
            continue
        a = np.asarray(tensors[k].data, dtype=np.float32)
        assert a.size == v.numel(), (k, a.shape, v.shape)
        new_sd[k] = torch.from_numpy(a.reshape(tuple(v.shape)).copy())
    model.load_state_dict(new_sd)
    model.model_config = cfg
    return model


REGISTRY["redimnet2_b6_mix_teacher"] = ("v2gguf", "OpenASR__redimnet2-b6-cn/redimnet2-b6-cn-fp16.oasr")


# ----------------------------------------------------------------------------
# EXTERNAL models -- NOT on the organisers' approved list. Used only for data
# cleaning / analysis (flagging mislabelled train/test files), never for the submission.
# ----------------------------------------------------------------------------
THREEDSPEAKER_SRC = ROOT / "third_party/3D-Speaker"


class FbankCMNWrap(nn.Module):
    """3D-Speaker front-end: Kaldi fbank (povey window, float wav) + per-utterance mean norm."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        feats = []
        for w in wav:
            f = torchaudio.compliance.kaldi.fbank(w.unsqueeze(0), num_mel_bins=80, sample_frequency=16000, dither=0.0)
            feats.append(f - f.mean(dim=0, keepdim=True))
        return self.backbone(torch.stack(feats))


def _load_3dspeaker(ckpt: Path, arch: str) -> nn.Module:
    sys.path.insert(0, str(THREEDSPEAKER_SRC))
    if arch == "eres2netv2":
        from speakerlab.models.eres2net.ERes2NetV2 import ERes2NetV2
        backbone = ERes2NetV2(feat_dim=80, embedding_size=192, baseWidth=26, scale=2, expansion=2)
    elif arch == "campplus":
        from speakerlab.models.campplus.DTDNN import CAMPPlus
        backbone = CAMPPlus(feat_dim=80, embedding_size=192)
    else:
        raise ValueError(arch)
    backbone.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))
    return FbankCMNWrap(backbone)


EXTERNAL = {
    "ext_wespeaker_r293_lm": ("wespeaker:avg_model.pt", "external/Wespeaker__wespeaker-voxceleb-resnet293-LM"),
    "ext_wespeaker_r221_lm": ("wespeaker:voxceleb_resnet221_LM.pt", "external/Wespeaker__wespeaker-voxceleb-resnet221-LM"),
    "ext_wespeaker_samr34_vb2": ("wespeaker:avg_model.pt", "external/gaunernst__wespeaker-voxblink2-samresnet34-ft"),
    "ext_3ds_eres2netv2_zh200k": ("3ds:eres2netv2", "external/3dspeaker/pretrained_eres2netv2.ckpt"),
    "ext_3ds_campplus_zh200k": ("3ds:campplus", "external/3dspeaker/campplus_cn_common.bin"),
}
REGISTRY.update(EXTERNAL)
