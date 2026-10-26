# VLSP 2026 SRED — hệ thống S5 + Q3 (đội greenearth03): gói inference để tái lập kết quả

Branch `minh-s5q3-final` chỉ chứa những thứ cần để **chạy inference và tái lập file kết quả private test** đã nộp:
code inference, trọng số model S5, tham số back-end Q3 và hướng dẫn này. Không có code train, không có dữ liệu.

| Thành phần | Nội dung |
|---|---|
| Model | **S5**: ReDimNet2-B6 nén low-rank, **6,917,480 tham số** (< 7M), 1 mạng duy nhất, embedding 192 chiều |
| Back-end | **Q3**: hồi quy logistic trên 9 đặc trưng của trial (cosine, độ dài, SNR), hệ số cố định trong `s5q3/assets/q3_qmf.json` |
| Kết quả tham chiếu | file nộp private `expected/sred_S5Q3_private_241026.zip` (278,410 trial) |
| Phần cứng đã kiểm chứng | NVIDIA L4 24 GB (CUDA 12.4), Python 3.12.9; chạy được cả trên CPU |

---

## 1. Cài đặt (~5 phút)

```bash
git clone -b minh-s5q3-final git@github.com:greenearth2003/VLSP-2026.git s5q3 && cd s5q3
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124   # GPU (CUDA 12.x)
# hoặc CPU: pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
md5sum -c MANIFEST.md5          # mọi dòng phải là OK
```

Yêu cầu:
- Python 3.12 và các gói trong `requirements.txt` (torch 2.5.1, torchaudio 2.5.1, numpy, scipy, soundfile).
- Không cần internet khi chạy: mã nguồn ReDimNet2 (MIT) nằm sẵn trong `third_party/redimnet2`, trọng số nằm trong
  `s5q3/assets/`.
- Không có checkpoint pretrained nào được tải về.

## 2. Kiểm tra nhanh (smoke test, ~1 phút)

```bash
python tests/smoke_test.py --audio-dir <thư mục private_test/data>
# OK: params 6,917,480; 40 files: min embedding cosine 0.99999.., durations and SNRs identical
```

Smoke test kiểm tra:
- md5 của 2 file tài sản;
- số tham số;
- với 40 file private bất kỳ: embedding (cosine ≥ 0.9999), độ dài và SNR trùng với pipeline lúc thi
  (`tests/expected_private_sample.npz`).

## 3. Chạy inference

```bash
python infer.py \
  --trials    <đường dẫn>/private_test/private-test.txt \
  --audio-dir <đường dẫn>/private_test/data \
  --out       out/private/results.txt \
  --device cuda --workers 4                         # --device cpu nếu không có GPU
cd out/private && zip ../sred_S5Q3_private.zip results.txt && cd -   # đúng định dạng nộp: zip chứa 1 file results.txt
```

**Đầu vào:**
- `--trials`: mỗi dòng `fileA fileB` hoặc `fileA fileB target|nontarget`. Đường dẫn tương đối với `--audio-dir`.
- Audio WAV 16 kHz. File nhiều kênh được lấy trung bình; sample rate khác 16 kHz sẽ báo lỗi.

**Đầu ra:**
- `results.txt`: mỗi dòng `fileA fileB score`, cùng thứ tự với file trial, score có 6 chữ số thập phân. Score càng
  cao thì càng có khả năng là cùng người nói. Đây đúng là định dạng nộp của VLSP 2026 SRED.
- Nếu file trial có nhãn, chương trình in thêm EER và minDCF(0.01).
- Embedding, độ dài và SNR được cache trong `<out>.cache.npz`, nên chạy lại chỉ mất vài giây để chấm lại điểm.

**Thời gian đo được:**
- NVIDIA L4: khoảng 62 ms/file. Private test (37,354 file, 278,410 trial) mất khoảng 40 phút, phần lớn là đọc audio
  và một lần forward cho mỗi file.
- CPU chậm hơn nhiều lần.

## 4. Tái lập kết quả đã nộp

```bash
unzip -o expected/sred_S5Q3_private_241026.zip -d expected/          # -> expected/results.txt (file đã nộp)
python tools/compare_results.py out/private/results.txt expected/results.txt
# lines 278,410 | pair-order mismatches 0 | identical lines ... | max |score diff| ...e-07 | ... => REPRODUCED
```

- **md5 của file đã nộp:** `results.txt` là `6998d3e02069c80608ca6dc193957d79`; file zip là
  `88f0b681b2bc96585af42695ad381f3c`.
- **Trên máy của chúng tôi**, chạy lại từ audio bằng gói này cho `results.txt` **giống hệt từng byte** (cùng md5
  `6998d3e0…`), xem mục 8. Kiểm tra nhanh:
  ```bash
  md5sum out/private/results.txt
  ```
- **Trên GPU, CPU hoặc phiên bản thư viện khác**, thứ tự cộng số thực có thể làm đổi chữ số thập phân thứ 6 của một
  vài điểm. Khi đó `compare_results.py` vẫn báo REPRODUCED (0 cặp lệch thứ tự, |diff| ≤ 1e-5), và EER không đổi.

**Kiểm tra trên public test** (có nhãn, chỉ để đối chiếu):
```bash
python infer.py --trials <...>/public_test/public-test.txt --audio-dir <...>/public_test/data --out out/public/results.txt
# EER 4.357%  minDCF(0.01) 0.2707   (raw cosine EER 4.613%)
```

## 5. Hệ thống chấm điểm một trial như thế nào

Mỗi trial (A, B) chỉ dùng **đúng 2 file audio của nó**. Không dùng thông tin từ trial hay file test nào khác:
không score normalization, không cohort test, không query expansion. Hệ thống **không transductive**.

1. **Đọc audio** (`s5q3/audio.py`): float32 trong [−1, 1], mono, 16 kHz.
2. **Đo chất lượng từng file** (không dùng model):
   - độ dài d = số mẫu / 16000, làm tròn 3 chữ số;
   - SNR ước lượng q = p95 − p10 của năng lượng khung 25 ms / 10 ms, với năng lượng = 10·log10(mean(x²) + 1e−10);
     làm tròn 2 chữ số.
3. **Embedding** (`s5q3/model.py`):
   - giữ 60 s đầu; file ngắn hơn 0.5 s thì lặp tín hiệu cho đủ 0.5 s;
   - **một** lần forward qua S5 trên cả file, rồi chuẩn hóa L2. Không cắt đoạn, không TTA.
4. **Cosine:** s = ⟨e_A, e_B⟩.
5. **Q3** (`s5q3/qmf.py`):
   - đặc trưng x = [s, log d_min, log d_max, s·log d_min, s·log d_max, lo, hi, s·lo, s·hi], với
     lo = clip(min(q_A, q_B), −5, 40)/10 và hi = clip(max(q_A, q_B), −5, 40)/10;
   - score = Σ_k coef_k · (x_k − mean_k)/std_k + intercept.
   - Các tham số được fit **chỉ trên dữ liệu train** của BTC và cố định trong `q3_qmf.json`. Lúc inference không fit
     gì cả.

## 6. Mô hình S5: tóm tắt nguồn gốc (chi tiết trong technical report)

- **Kiến trúc:** ReDimNet2-B6 (`b6-vb2+vox2+cnc2_v0-lm`, checkpoint trong danh sách BTC) được nén xuống 6,917,480
  tham số. 14 lớp chiếu 1×1 được phân rã low-rank theo activation (y = B(A x)); cấu trúc nằm trong `s5_model.pt`
  (`b6_cfg`, `ranks`, `drop`).
- **Huấn luyện**, chỉ dùng dữ liệu train BTC; bg_noise và nền yên lặng cắt từ chính bản ghi train chỉ dùng làm noise:
  1. distill từ teacher T1 (B6 fine-tune, 12.46M) → C2;
  2. thêm speaker, tách ID hai giọng, hard negatives, mô phỏng noise kiểu tập đánh giá → S2;
  3. **distill từ ensemble (T1 + S2)** trên cùng crop ồn → S5.
- **Teacher và ensemble chỉ dùng lúc train.** Gói này không chứa và không dùng chúng.
- **Kết quả nội bộ:**

  | Tập | S5 + Q3 | C2 + Q3 (bản nộp trước) |
  |---|---|---|
  | silver (public + noise), all / cleaned | 5.245 / 3.751 | 5.601 / 4.066 |
  | public, all / cleaned | 4.357 / 2.622 | 4.409 / 2.714 |

## 7. Cấu trúc thư mục

```
infer.py                      # trial list + audio -> results.txt (+ EER nếu có nhãn)
s5q3/model.py                 # dựng lại mạng S5 (ReDimNet2-B6 + LowRank), load trọng số, embed()
s5q3/audio.py                 # đọc wav, duration_s(), snr_est()
s5q3/qmf.py                   # back-end Q3 (hệ số cố định)
s5q3/assets/s5_model.pt       # trọng số + cấu trúc S5 (29 MB)
s5q3/assets/q3_qmf.json       # tham số Q3 của S5
third_party/redimnet2/        # mã nguồn ReDimNet2 (PalabraAI, MIT License), bản sao nguyên gốc
tests/smoke_test.py           # kiểm tra nhanh + tests/expected_private_sample.npz
tools/compare_results.py      # so file tái lập với file đã nộp
tools/export_assets.py        # (cần repo train) sinh 2 file tài sản từ checkpoint huấn luyện
tools/verify_against_repo.py  # (cần repo train) đối chiếu gói này với pipeline lúc thi
expected/sred_S5Q3_private_241026.zip   # file kết quả private đã nộp
MANIFEST.md5                  # md5 mọi file của gói
```

## 8. Kết quả kiểm chứng (26/10/2026, NVIDIA L4)

`tools/verify_against_repo.py` (so với repo huấn luyện):

| Kiểm tra | Kết quả |
|---|---|
| số tham số | 6,917,480 |
| mạng trong gói vs mạng trong repo, đầu vào ngẫu nhiên | chênh lệch tối đa 0.0 |
| 300 file private: embedding gói vs embedding repo | cosine nhỏ nhất 0.99999988 |
| độ dài / SNR vs thống kê repo | chênh lệch 0.0 / 0.0 |
| 278,410 trial: Q3 của gói (trên embedding đã cache) vs file đã nộp | **278,410 / 278,410 dòng giống từng byte** |

**Chạy lại end-to-end từ audio** (`infer.py`, 37,354 file, NVIDIA L4, khoảng 66 ms/file, khoảng 41 phút):

| Kiểm tra | Kết quả |
|---|---|
| số dòng | 278,410 |
| cặp lệch thứ tự | 0 |
| dòng giống từng byte | 278,410 (100 %) |
| md5 `results.txt` | `6998d3e02069c80608ca6dc193957d79`, **trùng file đã nộp** |

Số liệu public ở mục 4 (EER 4.357) được đo bằng pipeline lúc thi trên cùng model và back-end.

## 9. Giấy phép

- `third_party/redimnet2`: MIT License (PalabraAI), xem `third_party/redimnet2/LICENSE`.
- Trọng số S5 được huấn luyện từ checkpoint ReDimNet2-B6 do BTC cho phép, trên dữ liệu VLSP 2026 SRED.
