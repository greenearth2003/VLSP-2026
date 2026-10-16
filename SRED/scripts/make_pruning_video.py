"""Explainer animation: pruning the ReDimNet2-B6 teacher to fit the 7M budget (real numbers from exp/).

    /opt/conda/bin/python scripts/make_pruning_video.py exp/analysis/pruning_explainer.mp4
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

FPS = 30
W, H = 16, 9
BG, INK, MUTE, LINE = "#101318", "#ECEDEF", "#9AA0AA", "#2A2F38"
TEACH, STUD, KEEP, CUT, ERR, LIMIT = "#5B9BF8", "#F5A524", "#3CCB7F", "#4A505B", "#F06A6A", "#E8E8E8"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK})


def ease(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def seg(t, a, b):
    """Progress of t inside [a, b] (0..1), eased."""
    return ease((t - a) / max(1e-6, b - a))


def txt(ax, x, y, s, size=20, color=INK, alpha=1.0, ha="left", va="center", weight="normal"):
    if alpha > 0.01:
        ax.text(x, y, s, fontsize=size, color=color, alpha=float(np.clip(alpha, 0, 1)), ha=ha, va=va, weight=weight)


def box(ax, x, y, w, h, fc, ec=None, alpha=1.0, lw=1.5, r=0.08):
    if alpha > 0.01:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec or fc,
                                    lw=lw, alpha=float(np.clip(alpha, 0, 1))))


def arrow(ax, x0, y0, x1, y1, color=MUTE, alpha=1.0, lw=2):
    if alpha > 0.01:
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16, color=color, lw=lw,
                                     alpha=float(np.clip(alpha, 0, 1))))


def header(ax, t, title, sub=None):
    a = seg(t, 0, 0.08)
    txt(ax, 0.7, 8.3, title, 28, alpha=a, weight="bold")
    if sub:
        txt(ax, 0.7, 7.75, sub, 16, MUTE, alpha=a)


# ----------------------------------------------------------------------------- scenes
def s_title(ax, t):
    a = seg(t, 0, 0.25)
    txt(ax, 8, 5.2, "Pruning the teacher to fit the edge budget", 34, alpha=a, ha="center", weight="bold")
    txt(ax, 8, 4.3, "VLSP 2026 SRED  ·  ReDimNet2-B6 (12.5M)  →  ≤ 7M parameters", 20, MUTE, alpha=seg(t, 0.15, 0.4), ha="center")
    txt(ax, 8, 3.4, "structured pruning  +  least-squares repair  +  recovery distillation", 17, TEACH,
        alpha=seg(t, 0.3, 0.55), ha="center")


def s_problem(ax, t):
    header(ax, t, "The problem", "The best model is too big to submit")
    base, scale = 1.4, 0.42  # y of bars bottom, inches per M params
    bars = [("Teacher  ReDimNet2-B6", 12.46, TEACH, "EER 4.19 / 2.31 cleaned"),
            ("Student  ReDimNet-M (best, distilled)", 4.81, STUD, "EER 4.50 / 2.92 cleaned")]
    for i, (name, p, c, e) in enumerate(bars):
        x = 2.2 + i * 5.2
        g = seg(t, 0.1 + 0.12 * i, 0.4 + 0.12 * i)
        ax.add_patch(Rectangle((x, base), 2.6, p * scale * g, fc=c, alpha=0.9))
        txt(ax, x + 1.3, base + p * scale * g + 0.35, f"{p * g:.2f}M", 20, ha="center", alpha=g, weight="bold")
        txt(ax, x + 1.3, base - 0.35, name, 14, MUTE, ha="center", alpha=g)
        txt(ax, x + 1.3, base - 0.75, e, 13, c, ha="center", alpha=seg(t, 0.45, 0.6))
    la = seg(t, 0.5, 0.65)
    y7 = base + 7 * scale
    ax.plot([1.6, 9.9], [y7, y7], color=LIMIT, lw=2.5, ls="--", alpha=la)
    txt(ax, 10.05, y7, "7M limit", 17, LIMIT, alpha=la, weight="bold")
    txt(ax, 11.4, 6.3, "Teacher: 0.6 EER better,\nbut 1.8× over budget.", 17, alpha=seg(t, 0.65, 0.8))
    txt(ax, 11.4, 2.6, "Idea: keep its architecture,\nremove ~half of its channels.", 17, KEEP, alpha=seg(t, 0.8, 0.95))


STAGES = [(3, 0.730), (4, 0.730), (5, 0.995), (5, 0.995), (4, 1.560), (3, 2.171)]  # (#2D blocks, 1D block M)


def draw_net(ax, t, y=3.6, alpha=1.0, highlight_iface=0.0):
    x = 0.9
    for si, (n2, p1) in enumerate(STAGES):
        txt(ax, x + 0.9, y + 2.15, f"stage {si}", 12, MUTE, alpha=alpha, ha="center")
        for b in range(n2):
            box(ax, x + b * 0.32, y + 1.35, 0.26, 0.5, KEEP, alpha=0.85 * alpha, r=0.04)
        h1 = 0.35 + p1 * 0.55
        box(ax, x, y + 1.15 - h1, 1.75, h1, TEACH, alpha=0.85 * alpha, r=0.05)
        txt(ax, x + 0.87, y + 1.15 - h1 / 2, f"{p1:.2f}M", 12, BG, alpha=alpha, ha="center", weight="bold")
        if si < len(STAGES) - 1 and highlight_iface > 0:
            ax.plot([x + 2.05, x + 2.05], [y - 0.7, y + 2.0], color=ERR, lw=3, alpha=highlight_iface)
        x += 2.3
    box(ax, 14.75, y + 0.2, 0.9, 1.5, "#B07CF0", alpha=0.85 * alpha, r=0.05)
    txt(ax, 15.2, y + 0.95, "pool\n+\nemb", 11, BG, alpha=alpha, ha="center", weight="bold")


def s_anatomy(ax, t):
    header(ax, t, "Where B6's 12.5M parameters live", "6 stages · each = small 2D conv blocks + one big 1D time-context block")
    a = seg(t, 0.05, 0.25)
    draw_net(ax, t, alpha=a, highlight_iface=seg(t, 0.6, 0.72))
    b = seg(t, 0.25, 0.4)
    legend = [(TEACH, "1D time-context blocks", "7.18M  (58%)"), (KEEP, "2D residual conv blocks", "2.88M  (23%)"),
              ("#B07CF0", "pooling + embedding layer", "1.81M  (15%)")]
    for i, (c, n, v) in enumerate(legend):
        box(ax, 1.0, 1.6 - i * 0.55, 0.35, 0.35, c, alpha=b, r=0.04)
        txt(ax, 1.55, 1.77 - i * 0.55, n, 15, alpha=b)
        txt(ax, 5.3, 1.77 - i * 0.55, v, 15, MUTE, alpha=b)
    txt(ax, 8.2, 1.5, "Red lines: the C×F = 4608-wide interface between\nstages is shared by every block — it must stay intact.",
        15, ERR, alpha=seg(t, 0.65, 0.8))
    txt(ax, 8.2, 0.55, "So we only cut channels that live INSIDE a block.", 16, KEEP, alpha=seg(t, 0.82, 0.95), weight="bold")


def s_what(ax, t):
    header(ax, t, "What we cut: channels inside each block", "Example: stage 5's 1D time-context block (the biggest one)")
    a = seg(t, 0.05, 0.2)
    box(ax, 0.8, 3.0, 1.6, 2.6, LINE, alpha=a)
    txt(ax, 1.6, 4.3, "4608\nin", 15, alpha=a, ha="center")
    arrow(ax, 2.5, 4.3, 3.3, 4.3, alpha=a)
    txt(ax, 2.9, 4.75, "squeeze", 11, MUTE, alpha=a, ha="center")
    # channel dots: 192 -> keep 80
    n, keep = 192, 88
    rng = np.random.default_rng(3)
    kept = np.zeros(n, bool)
    kept[rng.choice(n, keep, replace=False)] = True
    cut = seg(t, 0.4, 0.65)
    cols, rows = 16, 12
    for i in range(n):
        cx, cy = 3.6 + (i % cols) * 0.22, 5.65 - (i // cols) * 0.22
        if kept[i]:
            ax.add_patch(plt.Circle((cx, cy), 0.075, color=KEEP if cut > 0.05 else TEACH, alpha=a))
        else:
            ax.add_patch(plt.Circle((cx, cy), 0.075, color=TEACH, alpha=a * (1 - 0.8 * cut)))
    txt(ax, 5.35, 2.65, f"bottleneck channels: 192 → {int(192 - (192 - keep) * cut)}", 15, KEEP if cut > 0.5 else INK,
        alpha=a, ha="center")
    txt(ax, 5.35, 2.2, "conv + attention mixers work in this narrow space", 12, MUTE, alpha=a, ha="center")
    arrow(ax, 7.2, 4.3, 8.0, 4.3, alpha=a)
    txt(ax, 7.6, 4.75, "expand", 11, MUTE, alpha=a, ha="center")
    box(ax, 8.1, 3.0, 1.6, 2.6, LINE, alpha=a)
    txt(ax, 8.9, 4.3, "4608\nout", 15, alpha=a, ha="center")
    ax.annotate("", xy=(8.9, 5.7), xytext=(1.6, 5.7), arrowprops=dict(arrowstyle="-|>", color=MUTE, lw=1.5,
                connectionstyle="arc3,rad=-0.25", alpha=a))
    txt(ax, 5.25, 7.2, "skip connection", 11, MUTE, alpha=a, ha="center")
    b = seg(t, 0.68, 0.85)
    txt(ax, 10.2, 5.6, "Same idea everywhere:", 17, alpha=b, weight="bold")
    for i, s in enumerate(["1D blocks: bottleneck  → ~45 %", "2D blocks: mid channels → ~50 %",
                           "head: 224 → 180 channels"]):
        txt(ax, 10.2, 4.9 - i * 0.55, "•  " + s, 14, alpha=seg(t, 0.72 + 0.05 * i, 0.85 + 0.05 * i))
    txt(ax, 10.2, 2.9, "Interfaces (4608) untouched →\nthe network still plugs together.", 14, KEEP, alpha=seg(t, 0.88, 0.98))


def s_choose(ax, t):
    header(ax, t, "Which channels to keep?", "Score = how much each channel actually contributes, measured on real audio")
    a = seg(t, 0.05, 0.2)
    txt(ax, 0.8, 6.7, "1. Run 1,024 random 3-s crops from the cleaned train set through B6", 16, alpha=a)
    txt(ax, 0.8, 6.15, "2. importance(channel) = ‖its output weights‖ × RMS of its activation", 16, alpha=seg(t, 0.12, 0.27))
    txt(ax, 0.8, 5.6, "3. Rank and keep the top-k", 16, alpha=seg(t, 0.22, 0.37))
    rng = np.random.default_rng(0)
    imp = np.sort(rng.gamma(1.6, 1.0, 64))[::-1]
    imp = imp / imp.max()
    k = 26
    g = seg(t, 0.3, 0.55)
    thr = seg(t, 0.55, 0.7)
    for i, v in enumerate(imp):
        c = KEEP if (i < k and thr > 0) else (CUT if thr > 0.5 else TEACH)
        ax.add_patch(Rectangle((0.9 + i * 0.13, 1.0), 0.1, v * 3.6 * g, fc=c, alpha=0.95))
    x_thr = 0.9 + k * 0.13 - 0.015
    ax.plot([x_thr, x_thr], [0.9, 4.8], color=INK, ls="--", lw=1.5, alpha=thr)
    txt(ax, x_thr + 0.1, 4.7, "keep ←  → cut", 13, alpha=thr)
    b = seg(t, 0.72, 0.85)
    box(ax, 10.2, 1.0, 5.2, 3.8, LINE, alpha=b)
    txt(ax, 10.5, 4.3, "Why data, not weights alone?", 16, alpha=b, weight="bold")
    txt(ax, 10.5, 3.5, "weight size only:", 14, MUTE, alpha=b)
    txt(ax, 10.5, 3.05, "agreement with teacher  0.18", 15, ERR, alpha=b)
    txt(ax, 10.5, 2.3, "activation-aware + repair:", 14, MUTE, alpha=seg(t, 0.8, 0.92))
    txt(ax, 10.5, 1.85, "agreement 0.63 (6.3M) · 0.78 (8.1M)", 15, KEEP, alpha=seg(t, 0.8, 0.92))
    txt(ax, 10.5, 1.3, "(cosine of embeddings, 60 public clips)", 11, MUTE, alpha=seg(t, 0.8, 0.92))


def s_repair(ax, t):
    header(ax, t, "Repair each block after cutting", "Refit the block's output layer by least squares so its output matches the original")
    a = seg(t, 0.03, 0.15)
    txt(ax, 0.8, 6.75, "original output  Y   ≈   W′ · H(kept channels)  +  b′          (solve for W′, b′ on the calibration audio)",
        16, alpha=a)
    txt(ax, 0.8, 6.15, "Done block by block, front to back — each block is repaired using the already-pruned blocks before it.",
        14, MUTE, alpha=seg(t, 0.1, 0.22))
    blocks = ["stage0", "stage1", "stage2", "stage3", "stage4", "stage5"]
    sliced = [0.315, 0.443, 0.611, 0.777, 0.622, 1.152]
    refit = [0.124, 0.209, 0.322, 0.308, 0.265, 0.167]
    ymax = 1.25
    sc = 4.2 / ymax
    for i, (b, s, r) in enumerate(zip(blocks, sliced, refit)):
        x = 1.3 + i * 2.2
        g1 = seg(t, 0.2 + 0.05 * i, 0.4 + 0.05 * i)
        g2 = seg(t, 0.55 + 0.04 * i, 0.72 + 0.04 * i)
        ax.add_patch(Rectangle((x, 1.0), 0.7, s * sc * g1, fc=ERR, alpha=0.85))
        ax.add_patch(Rectangle((x + 0.8, 1.0), 0.7, r * sc * g2, fc=KEEP, alpha=0.9))
        txt(ax, x + 0.35, 1.0 + s * sc * g1 + 0.2, f"{s:.2f}", 12, ERR, alpha=g1, ha="center")
        txt(ax, x + 1.15, 1.0 + r * sc * g2 + 0.2, f"{r:.2f}", 12, KEEP, alpha=g2, ha="center")
        txt(ax, x + 0.75, 0.6, b + " 1D", 12, MUTE, alpha=g1, ha="center")
    txt(ax, 14.6, 4.6, "relative\nerror", 13, MUTE, alpha=a, ha="center")
    box(ax, 13.9, 3.2, 0.3, 0.3, ERR, alpha=seg(t, 0.2, 0.3), r=0.03)
    txt(ax, 14.3, 3.35, "just cut", 13, alpha=seg(t, 0.2, 0.3))
    box(ax, 13.9, 2.6, 0.3, 0.3, KEEP, alpha=seg(t, 0.55, 0.65), r=0.03)
    txt(ax, 14.3, 2.75, "repaired", 13, alpha=seg(t, 0.55, 0.65))
    txt(ax, 8, 0.15, "Result: 12.46M → 6.33M — a working network, but far from B6 yet (zero-shot EER 14.8).", 16, KEEP,
        alpha=seg(t, 0.85, 0.97), ha="center", weight="bold")


def s_recover(ax, t):
    header(ax, t, "Recover with distillation from the full teacher", "18k training steps on the cleaned train set, all layers unfrozen")
    a = seg(t, 0.03, 0.18)
    box(ax, 0.6, 4.9, 3.3, 1.2, TEACH, alpha=a)
    txt(ax, 2.25, 5.5, "Teacher B6 (12.5M)\nhears full, clean audio", 12, BG, alpha=a, ha="center", weight="bold")
    box(ax, 0.6, 2.6, 3.3, 1.2, STUD, alpha=a)
    txt(ax, 2.25, 3.2, "Pruned B6 (6.33M)\nhears 0.75–3 s noisy crop", 12, BG, alpha=a, ha="center", weight="bold")
    arrow(ax, 4.0, 5.5, 4.9, 4.7, TEACH, alpha=a)
    arrow(ax, 4.0, 3.2, 4.9, 4.0, STUD, alpha=a)
    losses = ["match teacher's embedding", "match who-resembles-whom", "  in the batch",
              "speaker classifier, init", "  from teacher's centroids"]
    box(ax, 5.0, 2.9, 3.9, 2.6, LINE, alpha=seg(t, 0.15, 0.3))
    txt(ax, 6.95, 5.2, "training signals", 13, MUTE, alpha=seg(t, 0.15, 0.3), ha="center")
    for i, s in enumerate(losses):
        txt(ax, 5.2, 4.7 - i * 0.38, ("   " if s.startswith("  ") else "•  ") + s.strip(), 12,
            alpha=seg(t, 0.2 + 0.04 * i, 0.35 + 0.04 * i))
    # agreement curve
    steps = np.array([0, 6, 12, 18])
    agree = np.array([0.52, 0.805, 0.833, 0.842])
    x0, y0, w, h = 10.0, 1.4, 5.4, 4.6
    g = seg(t, 0.4, 0.8)
    ax.plot([x0, x0 + w], [y0, y0], color=LINE, lw=1.5, alpha=a)
    ax.plot([x0, x0], [y0, y0 + h], color=LINE, lw=1.5, alpha=a)
    sx = lambda s: x0 + s / 18 * w  # noqa: E731
    sy = lambda v: y0 + (v - 0.4) / 0.6 * h  # noqa: E731
    ax.plot([x0, x0 + w], [sy(1.0)] * 2, color=TEACH, ls=":", lw=1.5, alpha=a)
    txt(ax, x0 + w, sy(1.0) + 0.25, "= teacher", 12, TEACH, alpha=a, ha="right")
    n = max(2, int(2 + g * 60))
    xs = np.linspace(0, 18 * g, n)
    ys = np.interp(xs, steps, agree)
    ax.plot([sx(v) for v in xs], [sy(v) for v in ys], color=STUD, lw=3, alpha=a)
    for s, v in zip(steps, agree):
        if s <= 18 * g + 1e-6:
            ax.add_patch(plt.Circle((sx(s), sy(v)), 0.09, color=STUD))
            txt(ax, sx(s), sy(v) + 0.32, f"{v:.2f}", 12, alpha=1, ha="center")
    for s in steps:
        txt(ax, sx(s), y0 - 0.3, f"{s}k", 12, MUTE, alpha=a, ha="center")
    txt(ax, x0 + w / 2, y0 - 0.75, "training steps", 12, MUTE, alpha=a, ha="center")
    txt(ax, x0, y0 + h + 0.35, "agreement with teacher (public clips)", 13, MUTE, alpha=a)
    txt(ax, 0.8, 1.6, "Recovery is real but slows down: +0.29 in the first 6k steps,\nonly +0.04 in the last 12k.",
        15, alpha=seg(t, 0.82, 0.95))


def s_result(ax, t):
    header(ax, t, "Result and what we learned", "Full public test EER (%) — lower is better")
    rows = [("Teacher B6 · 12.5M (not allowed)", 4.19, TEACH), ("Distilled ReDimNet-M · 4.81M (current best)", 4.50, STUD),
            ("Pruned B6 · 6.33M, after recovery", 5.33, KEEP), ("Pruned B6 · 6.33M, right after pruning", 14.78, CUT)]
    for i, (n, v, c) in enumerate(rows):
        g = seg(t, 0.05 + 0.08 * i, 0.3 + 0.08 * i)
        y = 6.6 - i * 1.05
        vv = min(v, 8.0)
        ax.add_patch(Rectangle((6.2, y - 0.3), vv * 0.62 * g, 0.6, fc=c, alpha=0.9))
        txt(ax, 6.0, y, n, 14, alpha=g, ha="right")
        txt(ax, 6.2 + vv * 0.62 * g + 0.15, y, f"{v:.2f}" + ("  (off scale)" if v > 8 else ""), 14, alpha=g, weight="bold")
    b = seg(t, 0.55, 0.7)
    txt(ax, 0.8, 2.05, "Lesson: cutting half of B6 in one shot loses more than 18k steps of distillation can recover.", 16,
        ERR, alpha=b)
    txt(ax, 0.8, 1.35, "Next: prune progressively and recover in between —", 16, alpha=seg(t, 0.68, 0.8))
    for i, (s, c) in enumerate([("12.5M", TEACH), ("8.5M", KEEP), ("≤ 7M", STUD)]):
        g = seg(t, 0.75 + 0.06 * i, 0.85 + 0.06 * i)
        box(ax, 8.4 + i * 2.4, 1.0, 1.6, 0.7, c, alpha=g)
        txt(ax, 9.2 + i * 2.4, 1.35, s, 15, BG, alpha=g, ha="center", weight="bold")
        if i < 2:
            arrow(ax, 10.05 + i * 2.4, 1.35, 10.75 + i * 2.4, 1.35, alpha=seg(t, 0.8 + 0.06 * i, 0.9 + 0.06 * i))
    txt(ax, 8.4, 0.45, "prune  →  recover  →  prune  →  recover   (stage 1 running now)", 13, MUTE, alpha=seg(t, 0.88, 0.97))


SCENES = [(s_title, 4), (s_problem, 9), (s_anatomy, 10), (s_what, 11), (s_choose, 10), (s_repair, 10),
          (s_recover, 11), (s_result, 11)]


def main(out):
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 1])
    writer = FFMpegWriter(fps=FPS, codec="libx264", bitrate=4000, extra_args=["-pix_fmt", "yuv420p"])
    total = sum(d for _, d in SCENES) * FPS
    with writer.saving(fig, out, dpi=100):
        k = 0
        for fn, dur in SCENES:
            n = dur * FPS
            for f in range(n):
                ax.clear()
                ax.set_xlim(0, W)
                ax.set_ylim(0, H)
                ax.axis("off")
                ax.set_facecolor(BG)
                t = f / (n - 1)
                fn(ax, min(1.0, t * 1.15))  # finish the build early, hold the last ~13 %
                fade = min(1.0, f / 8, (n - 1 - f) / 8)  # fade in/out between scenes
                if fade < 1:
                    ax.add_patch(Rectangle((0, 0), W, H, fc=BG, alpha=1 - fade, zorder=100))
                # progress bar
                k += 1
                ax.add_patch(Rectangle((0, 0), W * k / total, 0.06, fc=TEACH, alpha=0.6, zorder=101))
                writer.grab_frame(facecolor=BG)
    print("wrote", out, f"({total / FPS:.0f} s)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pruning_explainer.mp4")
