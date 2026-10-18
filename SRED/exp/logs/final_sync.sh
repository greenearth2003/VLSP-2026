#!/bin/bash
# Lossless handoff sync: wait for M1/M2 evaluations, push git, upload new checkpoints + code snapshot + HANDOFF to HF,
# verify. Nothing is deleted.
set -u
R=/home/jovyan/work/thuc/VLSP-2026/SRED; S=/home/jovyan/work/thuc/hf_stage
cd $R
# (evaluations stopped by the user: sync immediately)


# ---- results summary for the handoff
.venv/bin/python - > exp/analysis/noise_robustness_summary.md <<'EOF'
import json
R = {}
for l in open("exp/results.jsonl"):
    r = json.loads(l); R[r.get("tag", "")] = r["results"]
rows = [("K21 (submitted; private LB 8.32)", "wise_K21_M", "noisy_K21", "realnoise_K21"),
        ("E2b SphereFace2", "wise_E2b_sf2", "noisy_E2b_sf2", "realnoise_E2b_sf2"),
        ("N10 SF2 + heavy generic noise", "wise_N10_sf2_noisy", "noisy_W_N10_sf2_noisy", "realnoise_W_N10_sf2_noisy"),
        ("N11 AAM + heavy generic noise", "wise_N11_aam_noisy", "noisy_W_N11_aam_noisy", "realnoise_W_N11_aam_noisy"),
        ("M1 SF2 + matched noise 50%", "wise_M1_p50", "noisy_W_M1_p50", "realnoise_W_M1_p50"),
        ("M2 SF2 + matched noise 80%", "wise_M2_p80", "noisy_W_M2_p80", "realnoise_W_M2_p80")]
f = lambda t: f"{R[t]['qmf_dur']['eer']:.2f}" if t in R else "n/a"
print("| system (WiSE 0.5, QMF dur) | clean public | noisy stand-in (synthetic) | noisy stand-in (real noise) |")
print("|---|---|---|---|")
for n, c, s, r in rows:
    print(f"| {n} | {f(c)} | {f(s)} | {f(r)} |")
EOF
cat exp/analysis/noise_robustness_summary.md

# ---- git: commit + push everything small (results, logs, scripts, analysis)
cd /home/jovyan/work/thuc/VLSP-2026
export GIT_CONFIG_GLOBAL=/home/jovyan/work/thuc/.gitconfig
export GIT_SSH_COMMAND="ssh -i /home/jovyan/work/thuc/.gitcredentials/id_ed25519_personal -o IdentitiesOnly=yes -F /dev/null"
git add SRED/exp/results.jsonl SRED/exp/logs SRED/exp/analysis/*.md SRED/exp/analysis/*.png SRED/exp/analysis/*.npy SRED/scripts SRED/sred
git commit -q -m "SRED handoff sync: N10/N11/M1/M2 noise-robustness results, logs, scripts, analysis

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01L8MbP6KxuCyGB7eN2FsJNn" || echo "nothing to commit"
git push -q origin thucth-private-test && echo "pushed $(git rev-parse --short HEAD)"

# ---- stage new checkpoints + fresh code snapshot
cd $R
for m in N10_sf2_noisy N11_aam_noisy M1_p50 M2_p80; do
  for f in exp/ft/W_$m/avg.pt exp/ft/$m/last.pt exp/ft/$m/step12000.pt exp/ft/$m/args.json; do
    [ -f $f ] && { mkdir -p $S/models/ft/$(dirname ${f#exp/ft/}); ln -f $f $S/models/ft/${f#exp/ft/}; }
  done
done
(cd /home/jovyan/work/thuc/VLSP-2026 && git archive --format=tar.gz -o $S/code/VLSP-2026_thucth-private-test.tar.gz --prefix=VLSP-2026/ thucth-private-test)
cp exp/analysis/noise_robustness_summary.md $S/
exit 0
