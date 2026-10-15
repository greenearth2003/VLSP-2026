# Tái lập hệ thống W01 — VLSP 2026 SRED

Tài liệu mô tả đầy đủ cách dựng lại hệ thống đã nộp, đạt **EER 4.85 % trên public test** (minDCF@0.01 ≈ 0.16–0.28 tuỳ nửa tập).
Hệ thống gồm 3 phần:

1. **Mô hình nhúng giọng nói W01** (ReDimNet-M, 4.81 M tham số), được tạo bằng cách
   *fine-tune ngắn* ReDimNet-M pretrained trên tập train VLSP (run **F03**), rồi
   *nội suy trọng số* 50/50 giữa model fine-tune và model pretrained (**WiSE-FT**).
2. **Trích embedding** từ toàn bộ file audio (không VAD, không cắt đoạn).
3. **Chấm điểm**: cosine giữa hai embedding đã chuẩn hoá L2, sau đó **hiệu chỉnh theo độ dài (QMF)**
   bằng một hồi quy logistic học trên dữ liệu *train* (không dùng nhãn test).

Toàn bộ chạy lại bằng một lệnh: `scripts/reproduce_w01.sh` (xem §8).

---

## Mục lục

1. [Môi trường](#1-môi-trường)
2. [Dữ liệu](#2-dữ-liệu)
3. [Mô hình gốc (pretrained)](#3-mô-hình-gốc-pretrained)
4. [Xử lý dữ liệu đầu vào](#4-xử-lý-dữ-liệu-đầu-vào)
5. [Huấn luyện (F03)](#5-huấn-luyện-f03)
6. [Nội suy trọng số WiSE-FT (W01)](#6-nội-suy-trọng-số-wise-ft-w01)
7. [Suy luận và chấm điểm](#7-suy-luận-và-chấm-điểm)
8. [Chạy lại từ đầu](#8-chạy-lại-từ-đầu)
9. [Kết quả và cách đánh giá](#9-kết-quả-và-cách-đánh-giá)
10. [Những gì đã thử nhưng không dùng](#10-những-gì-đã-thử-nhưng-không-dùng)
11. [Lưu ý về khả năng tái lập](#11-lưu-ý-về-khả-năng-tái-lập)

---

## 1. Môi trường

| Thành phần | Giá trị |
|---|---|
| GPU | 1× NVIDIA L4 (24 GB); đủ chạy với ~3 GB VRAM khi train, < 1 GB khi suy luận |
| Python | 3.12 |
| PyTorch / torchaudio | 2.5.1 + CUDA 12.4 |
| Quản lý package | `uv` (file `pyproject.toml`, `uv.lock`) |
| Thư viện chính | numpy < 2.2, scikit-learn, soundfile, safetensors, gguf |

```bash
cd /home/jovyan/code/VLSP-2026/SRED
uv sync                 # tạo .venv từ uv.lock
source env.sh           # LD_LIBRARY_PATH cho CUDA, TORCH_HOME, SRED_DATA
```

`env.sh`:

```bash
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
export TORCH_HOME=/home/jovyan/code/VLSP-2026/SRED/.cache/torch   # chứa mã nguồn ReDimNet (torch hub)
export PYTHONWARNINGS=ignore
export SRED_DATA=/home/jovyan/data/VLSP2026/Vi-SRED26
```

Mã nguồn kiến trúc ReDimNet (IDRnD) nằm ở `.cache/torch/hub/IDRnD_ReDimNet_master` (bản `master` của
`github.com/IDRnD/ReDimNet`). Nếu dựng môi trường mới, clone repo này vào đúng đường dẫn đó.

---

## 2. Dữ liệu

Thư mục `$SRED_DATA`:

| Tập | File | Nội dung |
|---|---|---|
| train | `train/train.txt`, `train/data/` | 58.404 câu, **1.900 người nói**, ~144 giờ. Mỗi dòng: `<speaker_id> <đường_dẫn_tương_đối>` |
| public test | `public_test/public-test.txt`, `public_test/data/` | 49.993 cặp thử (25.000 target), 17.850 file, 902 người nói. Mỗi dòng: `a.wav b.wav target\|nontarget` |

Đặc điểm audio: WAV PCM 16-bit, **16 kHz, mono**. Độ dài câu train: trung vị ~7.5 s, dài nhất ~28 s.
Public test có nhiều câu ngắn: 18 % số cặp có file ngắn nhất < 2 s, và đây là nơi tập trung lỗi (§9).

> Tên thư mục của public test chứa ID người nói. Hệ thống **không bao giờ dùng** thông tin này. Nó chỉ được dùng
> trong công cụ phân tích `sred/protocol.py` để chia tập theo người nói và tính khoảng tin cậy.

**Không dùng dữ liệu ngoài** (không MUSAN, không RIR thật, không VoxCeleb). Toàn bộ augmentation được tổng hợp
từ chính batch dữ liệu train (§4.3).

---

## 3. Mô hình gốc (pretrained)

**ReDimNet-M**, checkpoint `M-vb2+vox2+cnc-ft_mix` (thuộc danh sách pretrained được BTC cho phép).

- File: `pretrained/redimnet_v1/M-vb2+vox2+cnc-ft_mix.pt`, md5 `0a0acbbf1c4d6d558b1d796c4f464f8b`.
- Nguồn: bản phát hành chính thức của IDRnD trên GitHub. Trong môi trường này GitHub release bị chặn, nên file
  được tải từ HF mirror `voice-is-cool/ReDimNet` (cùng file).
- Đã huấn luyện trên VoxBlink2 + VoxCeleb2 + CN-Celeb, rồi fine-tune với dữ liệu trộn.
- **4.812 M tham số**, embedding 192 chiều.

Cấu hình (`model_config` trong checkpoint):

```python
{'C': 24, 'F': 72, 'embed_dim': 192, 'hop_length': 160, 'feat_type': 'pt',
 'block_1d_type': 'conv+att', 'block_2d_type': 'basic_resnet', 'pooling_func': 'ASTP',
 'global_context_att': True, 'group_divisor': 8, 'emb_bn': False, 'out_channels': None,
 'spec_params': {'do_preemph': True, 'do_spec_aug': True, 'norm_signal': True,
                 'freq_start_bin': 40, 'freq_mask_width': [0, 10], 'time_mask_width': [0, 20]},
 'stages_setup': [[1,4,2,[[3,3]],24], [2,5,2,[[3,3]],24], [1,6,2,[[3,3]],24],
                  [2,8,1,[[3,3]],24], [1,8,1,[[3,3]],24], [2,3,1,[[3,3]],24]]}
```

Cấu trúc backbone (theo thứ tự): `stem → stage0 … stage5 → fin_wght1d → mfa → fin_to2d`, sau đó là pooling ASTP
và lớp chiếu ra 192 chiều. Nạp model: `sred.models.load_model("redimnet_M_mix")`.

**Vì sao chọn ReDimNet-M?** Đánh giá zero-shot trên một tập con 1.500 cặp của public test, với điểm cosine:

| Model (≤ 7 M) | EER |
|---|---|
| **ReDimNet-M mix (4.81 M)** | **6.68** |
| ReDimNet-S mix (3.15 M) | 7.48 |
| WeSpeaker ResNet34-LM (6.63 M) | 10.01 |
| ReDimNet2-B4 vox2 (6.66 M) | 10.40 |

---

## 4. Xử lý dữ liệu đầu vào

### 4.1 Front-end đặc trưng (nằm trong model, áp dụng cho cả train và suy luận)

Model nhận trực tiếp **waveform float32 trong [-1, 1], 16 kHz**, dạng `[B, T]`. Phần `MelBanks` bên trong model
thực hiện các bước sau:

1. `NormalizeAudio`: chuẩn hoá biên độ tín hiệu (`norm_signal=True`).
2. Pre-emphasis với hệ số 0.97.
3. Mel spectrogram: `n_fft=512`, `win_length=400` (25 ms), `hop_length=160` (10 ms), `f_min=20`, `f_max=7600`,
   **72 mel bins**.
4. `log(x + 1e-6)`, sau đó chuẩn hoá trung bình theo trục thời gian (`spec_norm='mn'`).
5. **SpecAugment** (chỉ khi module ở chế độ `train`): mask tần số rộng 0–10 bin, mask thời gian rộng 0–20 frame.

Không có VAD, không cắt khoảng lặng, không resample: dữ liệu đã đúng 16 kHz mono. Khi đọc file, nếu có nhiều
kênh thì lấy trung bình các kênh, và chương trình dừng với lỗi nếu sample rate khác 16 kHz (`sred/data.py:load_wav`).

### 4.2 Lấy mẫu dữ liệu khi train

`sred/train.py`, các lớp `CropDataset` và `EndlessSampler`:

- **Sampler**: xáo trộn toàn bộ 58.404 câu, duyệt lần lượt rồi xáo lại (theo câu, không cân bằng theo người nói;
  `--spk-balanced` tắt). Hạt giống `seed=0`.
- **Crop 3 s ngẫu nhiên** cho mỗi câu khi đọc từ đĩa: chọn vị trí bắt đầu ngẫu nhiên và đọc đúng 48.000 mẫu.
  Câu ngắn hơn 3 s được lặp lại (`np.tile`) cho đủ 3 s.
- **Độ dài thay đổi theo batch** (`--min-crop-sec 1.5`): ở mỗi bước, toàn bộ batch được cắt tiếp xuống độ dài
  ngẫu nhiên trong [1.5, 3.0] s, làm tròn theo lưới 0.25 s để giới hạn số shape khác nhau cho cuDNN, tại một
  offset ngẫu nhiên. Mục đích là để model quen với câu ngắn, vì đây là chỗ public test sai nhiều nhất.
- Batch 32, `drop_last=True`, 7 worker DataLoader.
- Độ dài các câu train được cache sẵn trong `lists/train_lengths.json` (số mẫu của mỗi file).

### 4.3 Data augmentation trên GPU

`sred/augment.py`, lớp `Augmenter`. Augmentation chạy trên batch waveform trên GPU, mỗi mẫu được quyết định độc
lập, theo thứ tự sau:

| Bước | Xác suất | Chi tiết |
|---|---|---|
| Vang (reverb) | 0.3 | RIR tổng hợp: nhiễu Gauss nhân với hàm suy giảm mũ, RT60 ∈ U(0.2, 0.9) s, dài 0.6 s; giữ đường trực tiếp, đặt 0 trong 2.5 ms đầu; chuẩn hoá L2. Tích chập bằng FFT, sau đó đưa RMS về bằng RMS ban đầu. |
| Nhiễu cộng | 0.6 | Chọn đều một trong hai loại: **babble** (cộng 1–3 câu khác trong cùng batch, mỗi câu đã chuẩn hoá RMS) với SNR ∈ U(13, 20) dB; hoặc **nhiễu màu** 1/f^β với β ∈ U(0, 2) (từ trắng đến nâu) và SNR ∈ U(5, 20) dB. |
| Giới hạn băng tần | 0.2 | Mô phỏng điện thoại bằng đáp ứng kiểu Butterworth trong miền FFT: high-pass ở f_lo ∈ U(100, 400) Hz (bậc ~2), low-pass ở f_hi ∈ U(3000, 4000) Hz (bậc ~4). |
| EQ ngẫu nhiên | 0.3 | Mô phỏng màu âm của micro: 2 dải peaking dạng Gauss trên trục log-tần số, tâm f_c ∈ [150, 6000] Hz, độ rộng ∈ U(0.3, 1.2), gain ∈ U(−8, 8) dB; cộng thêm độ nghiêng ∈ U(−3, 3) dB/octave quanh 1 kHz. |
| μ-law | 0.1 | Lượng tử hoá μ-law 256 mức sau khi chuẩn hoá theo đỉnh (mô phỏng codec). |
| Chống clipping | luôn áp dụng | Mẫu nào có đỉnh > 1.0 thì được chia lại để đỉnh bằng 0.95; sau đó `nan_to_num`. |

Tham số dòng lệnh `--aug 1.0` nhân tất cả xác suất trên với 1.0, tức là dùng đúng giá trị trong bảng.
SpecAugment (§4.1) được áp dụng **thêm** sau các bước này, bên trong model.

---

## 5. Huấn luyện (F03)

### 5.1 Hàm mất mát: Sub-center AAM-softmax

`sred/train.py:SubCenterAAM`:

- **K = 3 sub-center** cho mỗi người nói, tổng cộng 1.900 × 3 vector trọng số 192 chiều. Cosine của một lớp
  = max cosine trên 3 sub-center của nó. Cách này giúp bền hơn với nhãn nhiễu và câu khác biệt trong cùng lớp.
- Additive angular margin **m = 0.2**, scale **s = 30**. Logit của lớp đúng là cos(θ + m). Khi θ + m > π thì
  dùng cos θ − m·sin m.
- Cross-entropy trên các logit đã nhân scale.

**Khởi tạo head từ tâm người nói.** Trước khi train, trích embedding của model pretrained cho 5 câu mỗi người nói
(`cohort_subset(per_spk=5, seed=0)`, tổng 9.500 câu, đọc toàn bộ câu). Trung bình các embedding đã chuẩn hoá của
từng người cho ra 1.900 tâm. Mỗi sub-center được khởi tạo bằng tâm của người nói đó cộng nhiễu N(0, 0.01²).
Nhờ vậy head đã gần đúng ngay từ đầu, và gradient ban đầu không phá các lớp backbone.

### 5.2 Phần nào được train

- **Đóng băng**: front-end `spec`, `stem`, `stage0`, `stage1`, `stage2` (tham số không có gradient, các module
  này để ở chế độ `eval`).
- **Train**: `stage3`, `stage4`, `stage5`, `fin_wght1d`, `mfa`, `fin_to2d`, pooling, lớp embedding, tổng
  **3.422 M / 4.812 M tham số**, cùng với head AAM.
- **Đóng băng BatchNorm** (`--freeze-bn`): ở **mỗi bước**, mọi lớp BatchNorm được chuyển về `eval`, nên dùng
  running mean/var của model pretrained và không cập nhật chúng. Trọng số affine γ/β của BN vẫn được học.
  Đây là thay đổi quan trọng giữa F01 và F03: batch nhỏ (32) và crop ngắn làm thống kê BN bị lệch.
- SpecAugment vẫn bật trong lúc train, vì module `spec` không bị đặt về `eval`.

### 5.3 Tối ưu hoá

| Siêu tham số | Giá trị |
|---|---|
| Optimizer | AdamW, weight decay 1e-4 cho backbone và 0 cho head |
| LR backbone | **2e-5** |
| LR head | 2e-3 |
| Lịch LR | warm-up tuyến tính 200 bước, sau đó cosine từ 1.0 về 0.05 × LR gốc ở bước cuối |
| Chỉ train head | 200 bước đầu (LR backbone = 0) |
| Tổng số bước | 3.000 (khoảng 96k mẫu, tương đương ~1.6 lượt qua dữ liệu train) |
| Batch | 32 |
| Mixed precision | bf16 autocast (front-end mel chạy fp32) |
| Gradient clipping | chuẩn L2 ≤ 5.0 |
| Seed | 0 (`torch.manual_seed`, `random.seed`) |
| TF32 | bật; `cudnn.benchmark=True` trong lúc train |

Lệnh gốc (`scripts/f03.sh`):

```bash
python -m sred.train --base redimnet_M_mix --name F03_M_s3_aug_bnfz_var --train-from stage3 \
  --bs 32 --crop-sec 3.0 --min-crop-sec 1.5 --amp bf16 --steps 3000 --lr 2e-5 --head-lr 2e-3 --warmup 200 \
  --head-only-steps 200 --eval-every 500 --dev-trials 2000 --aug 1.0 --freeze-bn
```

Diễn biến (log `exp/ft/F03_M_s3_aug_bnfz_var/log.txt`, khoảng 1.35 s/bước trên GPU dùng chung, tổng ~70 phút):

| Bước | Loss | Accuracy train | EER dev2000 (cosine) |
|---|---|---|---|
| 0 | – | – | 6.70 |
| 500 | 2.27 | 0.900 | 5.76 |
| 1000 | 2.18 | 0.898 | 5.66 |
| 1500 | 2.02 | 0.904 | 5.90 |
| **2000** | 1.59 | 0.933 | **5.56** ← `best.pt` |
| 2500 | 1.65 | 0.924 | 5.70 |
| 3000 | 1.55 | 0.927 | 5.56 |

**Checkpoint được dùng là bước 2000** (`exp/ft/F03_M_s3_aug_bnfz_var/best.pt` = `releases/version_111026_F03/sred_F03_redimnetM_111026.pt`, `step=2000`,
md5 `a5a46e78cf893727d193480decf0f77d`). Nó được chọn bằng 2.000 cặp ngẫu nhiên lấy từ public test (§11).

---

## 6. Nội suy trọng số WiSE-FT (W01)

Hệ thống cuối **không phải** F03 mà là trung bình tuyến tính trọng số của F03 và model pretrained:

```
θ_W01 = 0.5 · θ_F03(step 2000) + 0.5 · θ_ReDimNet-M pretrained
```

Phép trung bình áp dụng cho mọi tensor dạng float, bao gồm cả running mean/var của BN. Vì BN đã bị đóng băng
khi train, hai giá trị này trùng nhau giữa hai model. Các tensor nguyên như `num_batches_tracked` được lấy từ F03.

```bash
python scripts/wise_interp.py releases/version_111026_F03/sred_F03_redimnetM_111026.pt releases/version_141026_W01/sred_W01_redimnetM_141026.pt --alpha 0.5
```

Lệnh này tạo lại đúng `releases/version_141026_W01/sred_W01_redimnetM_141026.pt` (sai khác tối đa 0.0 so với file đã nộp).

- Số tham số vẫn là **4.812 M**, cùng kiến trúc với ReDimNet-M nên không tốn thêm chi phí suy luận.
- **Lý do dùng:** fine-tune làm model trôi khỏi lời giải pretrained. Thí nghiệm F05 (train dài gấp 3) cho thấy
  điều này làm hỏng các cặp dài. Nội suy trọng số giữ được phần lợi của fine-tune nhưng gần model gốc hơn.
  Kết quả là EER trên P-dev ngang F03, và đây là hệ thống duy nhất **vượt model zero-shot có ý nghĩa thống kê**
  (P = 0.98, §9). Giá trị α = 0.5 được chọn trước và chỉ thử một lần; không dò α trên test.

---

## 7. Suy luận và chấm điểm

### 7.1 Trích embedding

`sred/embed.py:embed_wav`, chế độ `full`:

- Đọc **toàn bộ** file và chạy một lượt forward, model ở chế độ `eval` (SpecAugment tắt, BN dùng running stats).
- Câu dài hơn 60 s được cắt còn 60 s đầu (public test không có câu nào dài như vậy).
- Câu ngắn hơn 0.5 s được lặp lại cho đủ 0.5 s.
- Embedding 192 chiều được **chuẩn hoá L2**.
- Không dùng VAD, không ghép nhiều đoạn crop, không test-time augmentation.

### 7.2 Điểm cơ sở

s = cosine(e_a, e_b), tức tích vô hướng của hai embedding đã chuẩn hoá L2. Không trừ trung bình, không dùng
AS-norm, LDA hay PLDA, vì tất cả đều đã thử và không tốt hơn (§10).

### 7.3 Hiệu chỉnh theo độ dài (QMF), backend `qmf_dur`

`sred/calibrate.py:QMF(use_asnorm=False)`. Lý do: các cặp ngắn có phân bố điểm bị nén, nên một ngưỡng chung cho
mọi cặp không tối ưu. Việc học lại thang điểm theo độ dài làm thay đổi thứ hạng gộp và giảm EER tổng.

**Đặc trưng** của mỗi cặp, với d_min và d_max là độ dài (giây) của file ngắn hơn và dài hơn:

```
x = [ s,  log d_min,  log d_max,  s·log d_min,  s·log d_max ]
```

**Dữ liệu học, chỉ lấy từ tập train, không dùng nhãn test:**

1. Cohort: 5 câu mỗi người nói × 1.900 người = 9.500 câu (`cohort_subset(per_spk=5, seed=0)`), trích embedding
   bằng **chính model W01**, đọc toàn bộ câu, chuẩn hoá L2. Độ dài lấy từ `lists/train_lengths.json`.
2. Cặp nontarget: 300.000 cặp chỉ số (i, j) ngẫu nhiên đều, dùng `numpy.default_rng(0)`.
3. Cặp target: tất cả các cặp cùng người nói trong cohort, khoảng 19.000 cặp (10 cặp × 1.900 người).
4. Loại cặp i = j. Nhãn của mỗi cặp là "cùng người nói hay không".
5. Chuẩn hoá z-score từng đặc trưng (mean/std tính trên tập học này).
6. `sklearn.linear_model.LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")`.

**Áp dụng cho test:** tính cùng 5 đặc trưng, trong đó độ dài file lấy bằng `soundfile.info(path).duration`.
Điểm nộp là `decision_function`, tức log-odds đã hiệu chỉnh: càng cao càng giống cùng người nói.

### 7.4 File nộp

`sred/evaluate.py:write_submission` ghi mỗi dòng theo định dạng `f"{a} {b} {score:.6f}"`, giữ nguyên thứ tự và
đường dẫn như trong file trial. Sau khi ghi, `scripts/make_submission.sh` kiểm tra:

- số dòng bằng file trial;
- từng cặp (a, b) trùng khớp đúng thứ tự;
- mỗi dòng có 3 trường và điểm là số hữu hạn.

Cuối cùng nén `results.txt` thành `results.zip` (dùng `zip -j`, để file nằm ở gốc archive).

```bash
scripts/make_submission.sh <trial_list> <audio_dir> <split_name> <out>/results.txt \
    releases/version_141026_W01/sred_W01_redimnetM_141026.pt qmf_dur
cd <out> && zip -j results.zip results.txt
```

Thời gian trên L4: ~20–25 phút cho 17.850 file public, cộng ~10 phút cho 9.500 câu cohort. Cả hai được cache
trong `exp/emb/<model>/`; cache được đặt tên theo tên split, nên muốn tính lại thì dùng tên split khác.

---

## 8. Chạy lại từ đầu

```bash
cd /home/jovyan/code/VLSP-2026/SRED && source env.sh
scripts/reproduce_w01.sh F03_repro
```

Script thực hiện:

1. Trích embedding cohort 9.500 câu của model pretrained, dùng để khởi tạo head (§5.1). Bước này cũng trích
   luôn public test để phân tích.
2. Train theo đúng công thức F03 (§5) với seed 0, **không dùng bất kỳ dữ liệu test nào**: lưu checkpoint mỗi
   500 bước và lấy thẳng `step2000.pt`.
3. Chạy `scripts/wise_interp.py` với α = 0.5 để tạo `exp/ft/W01_F03_repro/avg.pt`.
4. Suy luận, chấm điểm với `qmf_dur`, ghi `releases/version_<DDMMYY>_F03_repro/` (model `.pt`, `sred_F03_repro_public_<DDMMYY>.txt` và `.zip`).

Tổng thời gian khoảng 1.5–2 giờ trên một L4 dùng riêng (khoảng 0.5–0.7 s/bước train).

Để chỉ tái tạo file nộp từ checkpoint đã có, không train lại:

```bash
scripts/make_submission.sh $SRED_DATA/public_test/public-test.txt $SRED_DATA/public_test/data public_final \
    releases/version_DDMMYY_W01/sred_W01_public_DDMMYY.txt releases/version_141026_W01/sred_W01_redimnetM_141026.pt qmf_dur
```

---

## 9. Kết quả và cách đánh giá

### 9.1 Kết quả trên toàn bộ public test

| Hệ thống | Cosine | + QMF(dur) |
|---|---|---|
| ReDimNet-M zero-shot (E00) | 5.87 | 5.15 |
| F03 (fine-tune, bước 2000) | 5.31 | 4.88 |
| **W01 (nộp)** | – | **4.85** |

### 9.2 Quy trình chống overfit lên public test

Vì không có tập dev riêng, public test được chia theo người nói (hash md5 của ID) thành hai nửa rời nhau
(`sred/protocol.py`):

- **P-dev**: 228 người nói, 21.085 cặp. Mọi quyết định được đưa ra trên nửa này.
- **P-hold**: 236 người nói, 16.394 cặp. Chỉ mở **một lần** sau khi đã chốt hệ thống. Các cặp có hai người nói
  thuộc hai nửa khác nhau bị bỏ.

Quy tắc chấp nhận: hệ thống B chỉ thay hệ thống A khi bootstrap ghép cặp theo người nói (300 lần) cho
P(B tốt hơn) > 0.95 trên P-dev. Các thí nghiệm được đăng ký trước trong `EXPERIMENTS.md`.

| Hệ thống | P-dev QMF | P-hold QMF | P-hold minDCF | So với E00 trên P-hold |
|---|---|---|---|---|
| E00 zero-shot | 5.02 | 5.32 | 0.210 | – |
| F03 | 4.60 | 5.17 | 0.156 | −0.15 (P = 0.80) |
| **W01** | **4.55** | **5.15** | 0.161 | −0.17 (P = 0.94); cosine −0.39 (P = 0.99) |

Phần cải thiện vẫn giữ đúng chiều trên P-hold nhưng nhỏ hơn trên P-dev. Mức kỳ vọng thực tế trên private là
**tốt hơn zero-shot khoảng 0.2–0.5 điểm EER**, và minDCF tốt hơn rõ. Chênh lệch giữa các nhóm người nói lớn:
cùng một hệ thống cho 3.3 % ở nửa này và 5.5 % ở nửa kia, nên EER trên private có thể lệch hơn 1 điểm so với
public.

### 9.3 Lỗi theo độ dài (W01, P-dev, QMF; tính theo file ngắn hơn của cặp)

| d_min | < 2 s | 2–4 s | 4–8 s | ≥ 8 s |
|---|---|---|---|---|
| EER | 11.16 | 4.67 | 2.07 | 0.64 |

Câu ngắn dưới 2 s là nguồn lỗi chính.

---

## 10. Những gì đã thử nhưng không dùng

| Thử nghiệm | Kết quả | Kết luận |
|---|---|---|
| Back-end: trừ trung bình, AS-norm (k = 100/300/1000), LDA-128, WCCN, whitening, PLDA | Cosine 5.87 → tốt nhất 5.35 (LDA), còn PLDA là 6.39 | Đều kém QMF(dur) và dễ overfit; không dùng. |
| QMF thêm đặc trưng AS-norm hoặc độ lớn embedding | 5.15–6.50 | Có hại. Kết quả tốt hơn ở lần đầu là do lỗi rò rỉ cohort, đã sửa. |
| F01: như F03 nhưng không đóng băng BN, crop cố định 2 s, LR 5e-5 | Cosine 5.51 / QMF 5.10 | Kém F03. |
| F02: F01 + relational KD từ ReDimNet2-B6 | Dev xấu đi sau 1.000 bước | Dừng sớm. |
| F05: công thức F03 nhưng 9.000 bước, trung bình 3 checkpoint cuối | P-dev QMF 5.01, bằng zero-shot; cặp ≥ 8 s xấu đi (1.00 so với 0.57) | Train dài làm model trôi; bỏ. |
| K01: F05 + relational KD từ B6 (w = 10) | P-dev QMF 4.51 nhưng cosine 5.52, minDCF xấu hơn | Hoà W01, không chấp nhận. |
| Teacher ReDimNet2-B6 (12.5 M, không được nộp) | P-dev QMF 4.07 | Là trần tham khảo; KD chưa khai thác được. |
| Các model ≤ 7 M khác (S, ResNet34-LM, B4) | EER zero-shot 7.5–10.4 | Kém M rõ rệt. |

---

## 11. Lưu ý về khả năng tái lập

1. **Cách chọn checkpoint F03.** Lần chạy gốc chọn `best.pt` là bước 2000, dựa trên EER cosine của 2.000 cặp
   ngẫu nhiên (seed 0) lấy từ **toàn bộ** public test. Phiên bản hiện tại của `make_dev` chỉ lấy cặp từ P-dev,
   nên chạy lại `scripts/f03.sh` nguyên trạng có thể chọn checkpoint khác. Để tái lập đúng hệ thống đã nộp mà
   không đụng tới test, hãy dùng `scripts/reproduce_w01.sh`: script này lấy cố định `step2000.pt`.
2. **Không tất định ở mức bit.** `cudnn.benchmark`, TF32, bf16, và việc crop ngẫu nhiên trong các worker
   DataLoader làm trọng số sau khi train lại khác bản gốc một chút. Theo phân tích bootstrap, độ dao động EER
   cỡ ±0.2–0.3 là bình thường. Muốn có đúng file đã nộp thì dùng checkpoint gốc:
   `releases/version_141026_W01/sred_W01_redimnetM_141026.pt` (md5 `712e1923c0ffb6d44d5bf64d578915c1`).
3. **Cache embedding** trong `exp/emb/<run>_<stem>/` được đặt khoá theo `<thư mục run>_<tên file>` và tên split.
   Khi train lại với cùng tên run, cần xoá cache cũ.
4. **Nguồn pretrained.** Do GitHub release bị chặn trong môi trường này, checkpoint ReDimNet-M được tải từ HF
   mirror `voice-is-cool/ReDimNet` (cùng file). Nên xác nhận với BTC rằng nguồn này được chấp nhận.
5. **Container có thể khởi động lại** và giết các job đang chạy. `sred.train --resume` sẽ tiếp tục từ
   `resume.pt`, được lưu mỗi `--save-every` bước.

---

### Các file liên quan

| File | Vai trò |
|---|---|
| `sred/data.py` | Đọc danh sách train/trial, đọc wav, chọn cohort |
| `sred/augment.py` | Augmentation trên GPU (§4.3) |
| `sred/train.py` | Fine-tune: Sub-center AAM, đóng băng stage/BN, lịch LR, lưu/tiếp tục checkpoint, trung bình trọng số |
| `sred/models.py` | Nạp model pretrained và checkpoint fine-tune |
| `sred/embed.py` | Trích embedding, có cache |
| `sred/calibrate.py` | QMF theo độ dài (§7.3) |
| `sred/evaluate.py` | Chấm điểm, tính nhiều back-end, ghi file nộp |
| `sred/protocol.py` | Chia P-dev/P-hold, bootstrap ghép cặp |
| `scripts/wise_interp.py` | Nội suy trọng số WiSE-FT |
| `scripts/reproduce_w01.sh` | Chạy lại toàn bộ quy trình |
| `scripts/make_submission.sh` | Tạo và kiểm tra file nộp |
| `EXPERIMENTS.md` | Nhật ký số liệu đầy đủ |
