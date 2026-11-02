# VLSP 2026 SRED — hệ thống S5c + Q3 (đội greenearth03): gói inference để tái lập kết quả

Branch `minh-s5c-clean-final` chỉ chứa những thứ cần để **chạy inference và tái lập file kết quả private test** của
hệ thống S5c + Q3: code inference, trọng số model S5c, tham số back-end Q3 và hướng dẫn này. Không có code train, không
có dữ liệu.

> **Tóm tắt:**
> - S5c được train **chỉ trên dữ liệu train của BTC**; bg_noise chỉ được dùng làm noise.
> - **Không** có thông tin nào từ private test hay public test được dùng để train S2c / S5c hay để fit Q3.
> - Checkpoint S2c (step 6000) và S5c (step 2500) là **bước cuối cố định** của lịch train định sẵn. Không chọn
>   checkpoint.
> - Chấm điểm **không transductive**: mỗi trial chỉ dùng đúng 2 file của nó; không score normalization, không AS-norm,
>   không cohort test, không query expansion.
> - Model **6,917,480 tham số** (< 7M).
>
> Trên public test gốc (49,993 trial, đánh giá cuối):
>
> | Hệ thống | EER all | EER cleaned | minDCF(0.01) all | minDCF(0.01) cleaned |
> |---|---|---|---|---|
> | **S5c + Q3** (nhánh này) | **4.389 %** | **2.681 %** | **0.2689** | **0.2174** |
> | C2 + Q3 (bản nộp trước) | 4.409 % | 2.714 % | 0.2875 | 0.2364 |

| Thành phần | Nội dung |
|---|---|
| Model | **S5c**: ReDimNet2-B6 nén low-rank, **6,917,480 tham số** (< 7M), 1 mạng duy nhất, embedding 192 chiều |
| Back-end | **Q3**: hồi quy logistic trên 9 đặc trưng của trial (cosine, độ dài, SNR), hệ số cố định trong `s5c/assets/q3_qmf.json` |
| Kết quả tham chiếu | file kết quả private `expected/sred_S5cQ3_private_021126.zip` (278,410 trial) |
| Phần cứng đã kiểm chứng | NVIDIA L4 24 GB (CUDA 12.4), Python 3.12.9; chạy được cả trên CPU |

---

## 1. Cài đặt (~5 phút)

```bash
git clone -b minh-s5c-clean-final git@github.com:greenearth2003/VLSP-2026.git s5c && cd s5c
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124   # GPU (CUDA 12.x)
# hoặc CPU: pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
md5sum -c MANIFEST.md5          # mọi dòng phải là OK
```

Yêu cầu:
- Python 3.12 và các gói trong `requirements.txt` (torch 2.5.1, torchaudio 2.5.1, numpy, scipy, soundfile).
- Không cần internet khi chạy: mã nguồn ReDimNet2 (MIT) nằm sẵn trong `third_party/redimnet2`, trọng số nằm trong
  `s5c/assets/`.
- Không có checkpoint pretrained nào được tải về.

## 2. Kiểm tra nhanh (smoke test, ~1 phút)

```bash
python tests/smoke_test.py --audio-dir <thư mục private_test/data>
# OK: params 6,917,480; 40 files: min embedding cosine 0.9999999, durations and SNRs identical
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
cd out/private && zip ../sred_S5cQ3_private.zip results.txt && cd -   # đúng định dạng nộp: zip chứa 1 file results.txt
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
- NVIDIA L4: khoảng 68 ms/file. Private test (37,354 file, 278,410 trial) mất khoảng 43 phút, phần lớn là đọc
  audio và một lần forward cho mỗi file.
- CPU chậm hơn nhiều lần.

## 4. Tái lập file kết quả private

```bash
unzip -o expected/sred_S5cQ3_private_021126.zip -d expected/          # -> expected/results.txt (file tham chiếu)
python tools/compare_results.py out/private/results.txt expected/results.txt
# lines 278,410 | pair-order mismatches 0 | identical lines ... | max |score diff| ... | ... => REPRODUCED
```

- **md5 của file tham chiếu:** `results.txt` là `7ad1be9a8a1787f41ffbeb2353ed85ba`; file zip là
  `d8b95b42a23d6dcc4f0777abcecf56f3`. File này do pipeline huấn luyện tạo ra (cùng model, cùng back-end Q3).
- **Trên máy của chúng tôi**, chạy lại từ audio bằng gói này cho `results.txt` **giống hệt từng byte** (cùng md5
  `7ad1be9a…`), xem mục 8. Kiểm tra nhanh:
  ```bash
  md5sum out/private/results.txt
  ```
- **Trên GPU, CPU hoặc phiên bản thư viện khác**, thứ tự cộng số thực có thể làm đổi chữ số thập phân thứ 6 của một
  vài điểm. Khi đó `compare_results.py` vẫn báo REPRODUCED (0 cặp lệch thứ tự, |diff| ≤ 1e-5), và EER không đổi.

**Kiểm tra trên public test** (có nhãn, chỉ để đối chiếu):
```bash
python infer.py --trials <...>/public_test/public-test.txt --audio-dir <...>/public_test/data --out out/public/results.txt
# EER 4.389%  minDCF(0.01) 0.2689   (raw cosine EER 4.601%)
```

## 5. Hệ thống chấm điểm một trial như thế nào

Mỗi trial (A, B) chỉ dùng **đúng 2 file audio của nó**. Không dùng thông tin từ trial hay file test nào khác:
không score normalization, không AS-norm, không cohort test, không query expansion. Hệ thống **không transductive**.

1. **Đọc audio** (`s5c/audio.py`): float32 trong [−1, 1], mono, 16 kHz.
2. **Đo chất lượng từng file** (không dùng model):
   - độ dài d = số mẫu / 16000, làm tròn 3 chữ số;
   - SNR ước lượng q = p95 − p10 của năng lượng khung 25 ms / 10 ms, với năng lượng = 10·log10(mean(x²) + 1e−10);
     làm tròn 2 chữ số.
3. **Embedding** (`s5c/model.py`):
   - giữ 60 s đầu; file ngắn hơn 0.5 s thì lặp tín hiệu cho đủ 0.5 s;
   - **một** lần forward qua S5c trên cả file, rồi chuẩn hóa L2. Không cắt đoạn, không TTA.
4. **Cosine:** s = ⟨e_A, e_B⟩.
5. **Q3** (`s5c/qmf.py`):
   - đặc trưng x = [s, log d_min, log d_max, s·log d_min, s·log d_max, lo, hi, s·lo, s·hi], với
     lo = clip(min(q_A, q_B), −5, 40)/10 và hi = clip(max(q_A, q_B), −5, 40)/10;
   - score = Σ_k coef_k · (x_k − mean_k)/std_k + intercept.
   - Các tham số được fit **chỉ trên dữ liệu train** của BTC: cohort train sạch (cả file và 1.5 s đầu) + cùng cohort
     đó trộn với nửa train của bg_noise. Không dùng AS-norm. Tham số cố định trong `q3_qmf.json`; lúc inference không
     fit gì cả.

## 6. Mô hình S5c: tóm tắt nguồn gốc (chi tiết trong `docs/technical_report_S5cQ3.html`)

- **Kiến trúc:** ReDimNet2-B6 (`b6-vb2+vox2+cnc2_v0-lm`, checkpoint trong danh sách BTC) được nén xuống 6,917,480
  tham số. 14 lớp chiếu 1×1 được phân rã low-rank theo activation (y = B(A x)); cấu trúc nằm trong `s5c_model.pt`
  (`b6_cfg`, `ranks`, `drop`).
- **Huấn luyện**, chỉ dùng dữ liệu train BTC (bg_noise chỉ dùng làm noise):
  1. distill từ teacher T1 (ReDimNet2-B6 fine-tune trên train, 12.46M) → C2 → C2full (step 3000);
  2. **S2c**: khởi tạo từ C2full; KD tới T1 (T1 thấy crop sạch) + target embedding cả clip từ T1 và FF0
     (ReDimNet-M fine-tune trên train); dữ liệu = danh sách train đầy đủ (57,837 utterance), trong đó 43 ID có hai
     giọng được tách thành hai lớp; 6000 step, lr 3e-5;
  3. **S5c**: khởi tạo từ S2c; **distill từ ensemble (T1 + S2c)**, teacher thấy cùng crop đã augment như student;
     cùng target embedding và dữ liệu; 2500 step, lr 2e-5.
- **Augmentation khi train S2c / S5c:**
  - nền = các đoạn yên lặng cắt từ chính các bản ghi train của BTC, trộn với xác suất 0.8, SNR ~ U(0, 15) dB;
  - augmentation kênh chung: reverb, EQ, giới hạn băng tần, μ-law, noise nhẹ.
- **Checkpoint:** S2c step 6000 và S5c step 2500 là bước cuối cố định của lịch train định sẵn. Không chọn checkpoint.
- **Teacher và ensemble chỉ dùng lúc train.** Gói này không chứa và không dùng chúng.
- **Kết quả trên public test** (49,993 trial; cleaned: 47,932 trial):

  | Hệ thống | EER all | EER cleaned | minDCF(0.01) all | minDCF(0.01) cleaned |
  |---|---|---|---|---|
  | S5c + Q3 | 4.389 % | 2.681 % | 0.2689 | 0.2174 |
  | S5c, cosine thô (không Q3) | 4.601 % | 2.806 % | 0.2685 | – |
  | C2 + Q3 (bản nộp trước) | 4.409 % | 2.714 % | 0.2875 | 0.2364 |

## 7. Cấu trúc thư mục

```
infer.py                      # trial list + audio -> results.txt (+ EER nếu có nhãn)
s5c/model.py                  # dựng lại mạng S5c (ReDimNet2-B6 + LowRank), load trọng số, embed()
s5c/audio.py                  # đọc wav, duration_s(), snr_est()
s5c/qmf.py                    # back-end Q3 (hệ số cố định)
s5c/assets/s5c_model.pt       # trọng số + cấu trúc S5c (29 MB)
s5c/assets/q3_qmf.json        # tham số Q3 của S5c
third_party/redimnet2/        # mã nguồn ReDimNet2 (PalabraAI, MIT License), bản sao nguyên gốc
tests/smoke_test.py           # kiểm tra nhanh + tests/expected_private_sample.npz
tools/compare_results.py      # so file tái lập với file tham chiếu
tools/export_assets.py        # (cần repo train) sinh 2 file tài sản từ checkpoint huấn luyện
tools/verify_against_repo.py  # (cần repo train) đối chiếu gói này với pipeline lúc thi
expected/sred_S5cQ3_private_021126.zip   # file kết quả private tham chiếu
docs/technical_report_S5cQ3.html         # technical report (HTML)
MANIFEST.md5                  # md5 mọi file của gói
```

## 8. Kết quả kiểm chứng (02/11/2026, NVIDIA L4)

`tools/verify_against_repo.py` (so với repo huấn luyện):

| Kiểm tra | Kết quả |
|---|---|
| số tham số | 6,917,480 |
| mạng trong gói vs mạng trong repo, đầu vào ngẫu nhiên | chênh lệch tối đa 0.0 |
| 300 file private: embedding gói vs embedding repo | cosine nhỏ nhất 0.99999988 |
| độ dài / SNR vs thống kê repo | chênh lệch 0.0 / 0.0 |
| 278,410 trial: Q3 của gói (trên embedding đã cache) vs file tham chiếu | **278,410 / 278,410 dòng giống từng byte** |

**Chạy lại end-to-end từ audio** (`infer.py`, 37,354 file, NVIDIA L4, khoảng 68 ms/file, khoảng 43 phút):

| Kiểm tra | Kết quả |
|---|---|
| số dòng | 278,410 |
| cặp lệch thứ tự | 0 |
| dòng giống từng byte | 278,410 (100 %) |
| max \|score diff\| | 0.0 |
| md5 `results.txt` | `7ad1be9a8a1787f41ffbeb2353ed85ba`, **trùng file tham chiếu** |

Số liệu public ở mục 4 và mục 6 là của đánh giá cuối trên public test. Q3 của gói này, áp lên embedding public của
cùng model, cho đúng các số đó (EER 4.389 %, minDCF(0.01) 0.2689, cosine thô 4.601 %).

## 9. Giấy phép

- `third_party/redimnet2`: MIT License (PalabraAI), xem `third_party/redimnet2/LICENSE`.
- Trọng số S5c được huấn luyện từ checkpoint ReDimNet2-B6 do BTC cho phép, trên dữ liệu VLSP 2026 SRED.
