"""GPU twin of scripts/ext_vad_max.py `embed` (same per-clip outputs, same npz format), run with the separate onnxruntime-gpu
environment: /mnt/tsharp/thucth/dataset/vlsp2026-sred/.venv-ort/bin/python scripts/ext_vad_max_gpu.py --files list.txt --root DIR --out X.npz
Silero VAD and fbank on CPU in each worker, ResNet293 on the GPU (CUDAExecutionProvider). Analysis only.
"""
import argparse
import multiprocessing as mp

import numpy as np

ONNX = "/mnt/tsharp/thucth/dataset/vlsp2026-sred/external/hynt_Speaker-Embedding-30M/speaker_embedding_resnet293.onnx"
K, SR = 8, 16000
_s = {}


def _init():
    import onnxruntime as ort
    import torch
    from silero_vad import load_silero_vad
    torch.set_num_threads(1)
    ort.preload_dlls()
    o = ort.SessionOptions()
    o.log_severity_level = 3
    o.intra_op_num_threads, o.inter_op_num_threads = 1, 1
    # variable clip lengths: EXHAUSTIVE (default) re-benchmarks every conv for every new input shape, which is very slow
    _s["emb"] = ort.InferenceSession(ONNX, o, providers=[("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}),
                                                          "CPUExecutionProvider"])
    _s["vad"] = load_silero_vad(onnx=True)


def _emb(x16):
    import torch
    import torchaudio.compliance.kaldi as kaldi
    x = torch.from_numpy(x16)
    if len(x) < 400:
        x = x.repeat(int(np.ceil(400 / max(1, len(x)))))[:400]
    f = kaldi.fbank(x[None], num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0, sample_frequency=SR)
    e = _s["emb"].run(["embs"], {"feats": (f - f.mean(dim=0))[None].numpy()})[0][0]
    return e / np.linalg.norm(e)


def _process(path):
    import soundfile as sf
    import torch
    from silero_vad import get_speech_timestamps
    d, sr = sf.read(path, dtype="int16")
    x16 = d.astype(np.float32) if d.ndim == 1 else d.astype(np.float32).mean(1)
    ts = get_speech_timestamps(torch.from_numpy(x16 / 32768.0), _s["vad"], sampling_rate=SR, threshold=0.5,
                               min_speech_duration_ms=250, min_silence_duration_ms=300, speech_pad_ms=30)
    full = _emb(x16)
    sp = np.concatenate([x16[t["start"]:t["end"]] for t in ts]) if ts else x16[:0]
    vadcat = _emb(sp) if len(sp) >= SR // 2 else full
    segs = sorted([(t["end"] - t["start"], t["start"], t["end"]) for t in ts if t["end"] - t["start"] >= SR], reverse=True)[:K]
    S = np.zeros((K, 256), np.float32)
    for i, (_, a, b) in enumerate(segs):
        S[i] = _emb(x16[a:b])
    return full, vadcat, S, len(segs), sum(t["end"] - t["start"] for t in ts) / SR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    files = open(a.files).read().split()
    with mp.get_context("spawn").Pool(a.workers, initializer=_init) as pool:
        r = pool.map(_process, [f"{a.root}/{f}" for f in files], chunksize=8)
    np.savez(a.out, full=np.stack([x[0] for x in r]), vadcat=np.stack([x[1] for x in r]), segs=np.stack([x[2] for x in r]),
             n=np.array([x[3] for x in r]), speech=np.array([x[4] for x in r]))
    print(f"{len(files)} clips -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
