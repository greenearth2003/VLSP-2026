# Báo cáo hệ thống S04 + QS — VLSP 2026 SRED

Ngày: 16/10/2026 · Thư mục release: `releases/version_161026_S04qs/`

| Chỉ số | Giá trị |
|---|---|
| **EER public (49.993 trial)** | **4.784 %** (minDCF@0.01 = 0.298) |
| EER P-dev sạch (19.569 trial, đã bỏ ID nhãn lỗi) | 4.047 % (minDCF 0.280) |
| Số tham số khi suy luận | **4.812 M** (< 7 M) |
| Mô hình | ReDimNet-M (một model, không ensemble) |

So với các bản trước trên public: W01 4.85 → S04 4.82 → **S04 + QS 4.78**.

Hệ thống gồm 2 phần độc lập:

- **S04** — mô hình nhúng: ReDimNet-M pretrained, fine-tune ngắn trên tập train **đã loại các mẫu có nhãn lỗi**,
  sau đó nội suy trọng số WiSE-FT 50/50 với model pretrained.
- **QS** — calibration điểm: QMF theo độ dài, nhưng được fit trên cặp thử tổng hợp gồm cả câu đầy đủ **và các
  đoạn cắt 1.5 s** của chúng, để bộ calibration học được hành vi điểm số ở cặp ngắn.

Tái lập toàn bộ: `scripts/reproduce_s04qs.sh` (§7).

---

## 1. Dữ liệu và nguyên tắc

- Train: `train/train.txt` — 58.404 câu, 1.900 speaker, 145.9 giờ, WAV 16 kHz mono.
- Public test: 49.993 trial, 17.850 file, 902 speaker.
- **Không dùng dữ liệu ngoài** (Vietnam-Celeb đã thử ở D1/D2a và làm kết quả xấu đi — §6).
- **Không dùng bất kỳ thông tin nào của public test để huấn luyện hay chọn tham số**: mọi lựa chọn của S04 được
  quyết định trên tập validation V tách ra từ train; public (P-dev sạch) chỉ dùng để kiểm tra chiều kết quả.
- Toàn bộ việc lọc dữ liệu chỉ dùng train và embedding của model pretrained (chưa từng thấy nhãn VLSP).

---

## 2. Bước 1 — Làm sạch tập train (`lists/train_drop_bad.txt`)

Script: `eda/make_drop_bad_list.py` (tái tạo đúng từng dòng file đã dùng). Đầu vào: embedding ReDimNet-M pretrained
của **mọi** câu train (`eda/extract_train_emb.py`, chế độ cả câu, chuẩn hoá L2).

| Quy tắc | Định nghĩa | Số câu loại |
|---|---|---|
| ID chứa 2 giọng | speaker ≥ 8 câu; k-means 2 cụm (n_init=3, seed 0) trên embedding; cos giữa 2 tâm < 0.5 và cụm nhỏ ≥ 15 % → loại **toàn bộ** ID (không biết cụm nào là "chủ") | 2.708 (56 ID) |
| Nhãn đáng ngờ | cos tới tâm chính speaker đó (leave-one-out) < 0.35 **và** cos tới tâm một speaker khác > cos tới chính nó + 0.1 | 371 |
| File trùng giữa 2 ID | cos > 0.97 giữa hai câu thuộc hai ID khác nhau → loại cả hai | 51 |
| **Còn lại** | | **55.274 câu, 1.842 speaker** |

Lý do (từ EDA, `docs/version_141026_eda_report.md`): khoảng 4.6 % câu train thuộc các ID gộp 2 người (thường lệch
F0 ~100 Hz — một nam, một nữ). AAM-softmax bị ép kéo hai giọng khác nhau về cùng một lớp → nhãn nhiễu có hại.

---

## 3. Bước 2 — Loại tập validation V khỏi train (`--exclude-val`)

V = 300 speaker train (chọn ngẫu nhiên seed 0 trong số speaker ≥ 6 câu; `sred/valset.py`, `lists/val_v4.json`),
dùng để so sánh công thức huấn luyện mà không đụng tới public test. S04 được train **không có** các speaker này:
55.274 → **45.241 câu** thực sự dùng để train.

> Ghi chú: bản S04full (cùng công thức, train cả 55.274 câu) đã được thử và cho kết quả tương đương
> (public 4.82, P-dev sạch 4.12) — không tốt hơn, nên bản nộp giữ S04 gốc.

---

## 4. Bước 3 — Fine-tune mô hình S04

### 4.1 Mô hình gốc
ReDimNet-M `M-vb2+vox2+cnc-ft_mix` (danh sách pretrained được BTC cho phép), `pretrained/redimnet_v1/M-vb2+vox2+cnc-ft_mix.pt`
(md5 `0a0acbbf1c4d6d558b1d796c4f464f8b`), 4.812 M tham số, embedding 192 chiều. Front-end nằm trong model:
log-Mel 72 băng (n_fft 512, cửa sổ 25 ms, bước 10 ms, 20–7600 Hz), pre-emphasis, chuẩn hoá trung bình theo thời gian,
SpecAugment (chỉ khi train).

### 4.2 Dữ liệu đầu vào khi train (`sred/train.py`)
- Sampler: xáo trộn theo câu (không cân bằng speaker), seed 0.
- Mỗi câu: đọc một đoạn **3 s ngẫu nhiên** (câu ngắn hơn được lặp cho đủ).
- Mỗi batch: cắt tiếp về độ dài ngẫu nhiên **1.5–3.0 s** (lưới 0.25 s), cùng offset ngẫu nhiên.
- Augmentation trên GPU (`sred/augment.py`, profile mặc định): vang tổng hợp (p 0.3, RT60 0.2–0.9 s), nhiễu cộng
  (p 0.6: babble từ chính batch SNR 13–20 dB hoặc nhiễu màu 1/f^β SNR 5–20 dB), giới hạn băng (p 0.2), EQ ngẫu nhiên
  (p 0.3, ±8 dB), μ-law (p 0.1); + SpecAugment trong model.

### 4.3 Hàm mất mát và khởi tạo
- **Sub-center AAM-softmax**: K = 3 sub-center/speaker, margin 0.2, scale 30.
- Head (1.842 × 3 vector) khởi tạo bằng **tâm speaker** tính từ embedding pretrained (5 câu/speaker) + nhiễu nhỏ.

### 4.4 Phần được huấn luyện và tối ưu
| Thiết lập | Giá trị |
|---|---|
| Đóng băng | front-end, `stem`, `stage0–2`; **BatchNorm đóng băng** (dùng running stats của pretrained) |
| Huấn luyện | `stage3–5`, attention 1D, pooling, lớp embedding = 3.422 M tham số + head AAM |
| Optimizer | AdamW, weight decay 1e-4 (head: 0) |
| LR | backbone 2e-5, head 2e-3; warm-up 200 bước; cosine về 5 % |
| Chỉ train head | 200 bước đầu |
| Số bước / batch | 3.000 bước × 32 (khoảng 2.1 lượt qua 45k câu) |
| Khác | bf16, clip gradient 5.0, seed 0, 4 worker DataLoader |
| Checkpoint dùng | **bước 2000** (cố định trước, không chọn theo test) |

Diễn biến (`exp/ft/S04_dropbad/log.txt`): loss 2.47 (bước 200) → 1.82 (1000) → 1.35 (2000) → 1.26 (3000);
accuracy train 0.89 → 0.94; ~0.4 s/bước trên L4 (~20 phút).

Lệnh (tương đương `scripts/reproduce_s04qs.sh` bước 4):
```bash
python -m sred.train --base redimnet_M_mix --train-from stage3 --bs 32 --crop-sec 3.0 --min-crop-sec 1.5 --amp bf16 \
  --steps 3000 --lr 2e-5 --head-lr 2e-3 --warmup 200 --head-only-steps 200 --dev-trials 0 --eval-every 3000 \
  --save-every 500 --aug 1.0 --freeze-bn --seed 0 --workers 4 --exclude-val \
  --train-list lists/train_drop_bad.txt --name S04_dropbad
```

### 4.5 WiSE-FT
```
θ_S04 = 0.5 · θ(bước 2000) + 0.5 · θ(ReDimNet-M pretrained)
```
`python scripts/wise_interp.py exp/ft/S04_dropbad/step2000.pt exp/ft/W_S04_dropbad/avg.pt --alpha 0.5`.
Giữ lợi ích fine-tune nhưng hạn chế trôi khỏi lời giải pretrained (train lâu hơn — F05 — làm xấu kết quả).
Model nộp: `releases/version_161026_S04qs/sred_S04qs_redimnetM_161026.pt` (md5 `1303a0e772640a0380d750df898e2408`,
trùng với `releases/version_151026_S04/`).

---

## 5. Bước 4 — Chấm điểm với QS calibration

Script: `scripts/make_submission_qs.py`.

1. **Embedding**: mỗi file test → một lượt forward trên **toàn bộ câu** (không VAD, không cắt; < 0.5 s thì lặp cho đủ
   0.5 s; > 60 s thì cắt) → chuẩn hoá L2.
2. **Điểm thô**: s = cosine(e_a, e_b).
3. **QMF theo độ dài**: x = [s, log d_min, log d_max, s·log d_min, s·log d_max] (d = độ dài file, giây) → chuẩn hoá
   z-score → hồi quy logistic (C = 1, class_weight = balanced). Điểm nộp = log-odds (càng cao càng giống cùng người).
4. **QS — dữ liệu fit QMF** (điểm mới so với W01/S04):
   - Cohort train: 5 câu/speaker × 1.900 speaker = 9.100 câu (`cohort_subset(seed=0)`), embedding bằng chính model S04.
   - **Thêm bản cắt 1.5 s đầu** của cùng 9.100 câu (độ dài tương ứng = min(d, 1.5)) → 18.200 embedding.
   - Cặp nontarget: 300.000 cặp ngẫu nhiên; cặp target: mọi cặp cùng speaker trong cohort,
     **trừ cặp giữa một câu và chính bản cắt của nó** (tránh target "dễ giả").
   - Lý do: cohort gốc có trung vị 7 s, chỉ ~9 % < 2 s, trong khi lỗi tập trung ở cặp ngắn. Thêm bản cắt giúp
     QMF học đúng thang điểm của cặp ngắn–ngắn / ngắn–dài, nên điểm giữa các nhóm độ dài được căn chỉnh tốt hơn
     khi gộp chung một ngưỡng.

Định dạng nộp: mỗi dòng `file1.wav file2.wav score`, đúng thứ tự file trial, nén `results.txt` trong zip.

---

## 6. Kết quả và các phương án đã so sánh

### 6.1 Bảng so sánh

| Hệ thống | EER public | P-dev sạch EER | P-dev sạch minDCF | Ghi chú |
|---|---|---|---|---|
| ReDimNet-M zero-shot + QMF | 5.15 | 4.54 | 0.347 | mốc |
| W01 (F03 bước 2000 + WiSE) | 4.85 | 4.09 | 0.274 | bản nộp trước |
| S04 (QMF thường) | 4.82 | 4.09 | 0.291 | |
| **S04 + QS** | **4.78** | **4.05** | **0.280** | **bản nộp** |
| W01 + QS | – | 4.04 | 0.271 | QS cũng giúp W01 (−0.05) |
| S02full (nhất quán ngắn–dài) | 4.86 | 4.11 | 0.306 | không hơn |
| S04full (train cả V) | 4.82 | 4.12 | 0.290 | tương đương |
| D1vc / D2a (+ Vietnam-Celeb) | 4.94 / 4.94 | 4.16 / 4.24 | 0.339 / 0.329 | xấu hơn (lệch domain) |

P-dev sạch = nửa public theo speaker (hash md5), bỏ trial chạm tới 42 ID chứa 2 giọng và các cặp ID trùng người.

### 6.2 Theo độ dài (P-dev sạch, file ngắn hơn của cặp)

| d_min | < 2 s | 2–4 s | 4–8 s | ≥ 8 s |
|---|---|---|---|---|
| W01 | 10.64 | 3.89 | 1.67 | 0.44 |
| S04 | 10.70 | 4.03 | 1.59 | 0.22 |
| **S04 + QS** | 10.66 | 3.98 | 1.56 | 0.25 |

### 6.3 Đánh giá trung thực
- So với W01, S04 + QS tốt hơn ~0.05–0.07 EER cả trên public lẫn P-dev sạch, nhưng **chưa vượt ngưỡng ý nghĩa
  thống kê** (bootstrap theo speaker: −0.04 [−0.11, +0.14], P(better) = 0.38). Mức dao động khi train lại cùng
  công thức là ±0.15.
- Ưu điểm chắc chắn hơn: không có thiên lệch chọn checkpoint trên public (W01 từng chọn bước 2000 bằng public),
  dữ liệu train sạch hơn.
- Nhóm cặp < 2 s (~10.6 %) không cải thiện bằng bất kỳ phương pháp nào đã thử — giới hạn thông tin của audio ngắn.
- Khoảng 1.1 điểm EER trên public đến từ nhãn lỗi của chính tập test (ID chứa 2 giọng): EER W01 bỏ các target trial
  đó = 3.73 %.
- EER private có thể lệch > 1 điểm so với public do thành phần speaker (hai nửa public: 3.3 % so với 5.5 %).

---

## 7. Tái lập

Toàn bộ từ đầu (~2 giờ trên 1 × L4):
```bash
cd /home/jovyan/code/VLSP-2026/SRED && source env.sh
scripts/reproduce_s04qs.sh S04_repro
```
Các bước: embedding pretrained toàn bộ train → `train_drop_bad.txt` → tạo V → fine-tune → WiSE → QS scoring → zip.
Huấn luyện lại không tất định từng bit (cuDNN benchmark, bf16, crop ngẫu nhiên trong worker) — dao động EER ±0.15.
Để có đúng file đã nộp, dùng checkpoint gốc (md5 ở §4.5).

Chỉ tạo file nộp từ checkpoint có sẵn (đã kiểm chứng tái tạo giống hệt từng dòng file release, EER 4.784):
```bash
.venv/bin/python scripts/make_submission_qs.py <trials.txt> <audio_dir> <split_name> <out>/results.txt \
    releases/version_161026_S04qs/sred_S04qs_redimnetM_161026.pt
cd <out> && zip -j results.zip results.txt
```

### File liên quan
| File | Vai trò |
|---|---|
| `eda/extract_train_emb.py` | embedding pretrained cho toàn bộ train |
| `eda/make_drop_bad_list.py` | tạo `lists/train_drop_bad.txt` |
| `sred/valset.py` | tập validation V (`--exclude-val`) và đánh giá trên V |
| `sred/train.py`, `sred/augment.py` | fine-tune, augmentation |
| `scripts/wise_interp.py` | WiSE-FT |
| `sred/calibrate.py` | QMF (tham số `fid` để loại cặp câu–bản cắt của chính nó) |
| `scripts/make_submission_qs.py` | chấm điểm + QS calibration + ghi file nộp |
| `scripts/reproduce_s04qs.sh` | chạy lại toàn bộ |
| `releases/version_161026_S04qs/` | model, file điểm, zip |
| `EXPERIMENTS.md` | nhật ký toàn bộ thí nghiệm |

| md5 | File |
|---|---|
| `1303a0e772640a0380d750df898e2408` | `sred_S04qs_redimnetM_161026.pt` |
| `adb8455571998a1287b8ffe0dd7f25d3` | `sred_S04qs_public_161026.txt` |
| `042e9e5a2fb4a5472d402a3a214badc2` | `sred_S04qs_public_161026.zip` |
