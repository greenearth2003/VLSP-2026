# VLSP 2026 SRED — các hệ thống nộp bài và nhánh code tương ứng (đội greenearth03)

Repo có nhiều nhánh. File này cho biết **nhánh nào tạo ra file nộp nào**, kết quả của mỗi hệ thống và cách chạy lại.

## 1. Tóm tắt

| Nhánh | Hệ thống | Tham số inference | Valid (silver), all / cleaned | Public test, all / cleaned | File nộp private |
|---|---|---|---|---|---|
| [`minh-s5q3-final`](../../tree/minh-s5q3-final) | **S5 + Q3** | 6,917,480 | **5.245 % / 3.751 %** | **4.357 % / 2.622 %** | `sred_S5Q3_private_241026.zip` |
| [`thucth-distill-noiseaug-durationaware`](../../tree/thucth-distill-noiseaug-durationaware) | **E8s5000_wise0.7** (VE v1.0.lite) + calibrator | ≈ 6.95M | **5.109 % / 3.764 % | chưa đo | `result-private-E8s5000_wise0.7.zip` |

Các thước đo:
- EER (%); số thấp hơn là tốt hơn.
- **Valid** = tập *silver*: audio public test cộng noise nền thật (cắt từ bản ghi train) và rumble tổng hợp. Thước đo
  chính là EER trên toàn bộ 49,993 trial (*all*). *Cleaned* là 47,932 trial, đã bỏ các file sai nhãn hoặc không có
  tiếng nói.
- **Public test** = public test gốc, không thêm noise, 49,993 trial.
- Cả hai tập chỉ dùng để **đánh giá**. Không fit hay train gì trên chúng.

## 2. Nhánh `minh-s5q3-final`: S5 + Q3

- **Tạo ra file nộp:** `sred_S5Q3_private_241026.zip`.
  - md5 file zip: `88f0b681b2bc96585af42695ad381f3c`
  - md5 `results.txt` bên trong: `6998d3e02069c80608ca6dc193957d79`
  - 278,410 trial
- **Nội dung nhánh** (nhánh chỉ dành cho inference):
  - code inference;
  - trọng số model `s5q3/assets/s5_model.pt` (29 MB, nằm luôn trong git);
  - tham số back-end `s5q3/assets/q3_qmf.json`;
  - file nộp gốc `expected/sred_S5Q3_private_241026.zip`;
  - `README.md` hướng dẫn chi tiết.
- **Model:**
  - ReDimNet2-B6 nén low-rank xuống < 7M tham số;
  - huấn luyện: distill từ teacher T1 → thêm dữ liệu, hard negatives, mô phỏng noise kiểu tập valid (SilverSim) →
    **distill từ ensemble (T1 + S2)**;
  - back-end Q3: hồi quy logistic trên cosine, độ dài và SNR, fit chỉ trên dữ liệu train.
- **Tính hợp lệ:** mỗi trial chỉ được chấm từ đúng 2 file audio của nó, không transductive.
- **Kết quả:**

  | Tập | all | cleaned | minDCF(0.01) all |
  |---|---|---|---|
  | Valid (silver) | 5.245 | 3.751 | 0.364 |
  | Public test | 4.357 | 2.622 | 0.271 |

- **Chạy lại** (khoảng 40 phút trên 1 GPU NVIDIA L4):
  ```bash
  git clone -b minh-s5q3-final git@github.com:greenearth2003/VLSP-2026.git s5q3 && cd s5q3
  python3.12 -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
  md5sum -c MANIFEST.md5
  python infer.py --trials <private_test>/private-test.txt --audio-dir <private_test>/data --out out/private/results.txt
  md5sum out/private/results.txt        # 6998d3e02069c80608ca6dc193957d79 = file đã nộp
  ```
  Đã kiểm chứng: chạy lại từ audio cho `results.txt` **giống hệt từng byte** với file đã nộp. Chi tiết trong
  `README.md` của nhánh.

## 3. Nhánh `thucth-distill-noiseaug-durationaware`: E8s5000_wise0.7

- **Tạo ra file nộp:** `result-private-E8s5000_wise0.7.zip`.
- **Model** (VE v1.0.lite):
  - ReDimNet2-B6 fine-tune trên crop ngắn/dài có noise, rồi nén low-rank có whitening theo activation (≈ 6.95M);
  - phục hồi bằng KD từ model chưa nén (run E5, sau đó restart E8, step 5000);
  - WiSE trộn trọng số 0.7;
  - calibrator chất lượng (QMF2: norm + speech view + AS-norm), fit trên speaker của tập train.
- **Kết quả valid (silver):** 5.109 / 3.764 (all / cleaned), theo `REPORT.md` của nhánh.
- **Public test:** chưa đo trong repo này.
- **Tài liệu và code:** `SRED/experiments/distill_noiseaug_durationaware/` trong nhánh: `REPORT.md`, `README.md`,
  `HANDOFF.md`.
- **Inference:** `ve_infer.py score --model-dir MODEL_DIR --trials ... --audio-dir ... --out results.txt`.
  - `MODEL_DIR` gồm `ve_v1.0.lite.pt`, `calibrator.json`, `calibrator.cohort.npy`, `b6_template.pt`.
  - Các file này nằm trong Hugging Face dataset riêng `pieceofcakee/vlsp2026`, không nằm trong git.

## 4. Ghi chú chung

- Ràng buộc của cuộc thi:
  - model inference < 7M tham số;
  - chỉ dùng dữ liệu BTC (bg_noise chỉ dùng làm noise);
  - không chấm điểm transductive;
  - nhãn public test không dùng để tối ưu model.
- Định dạng nộp: file zip chứa đúng một file `results.txt`; mỗi dòng `fileA fileB score`, cùng thứ tự với
  `private-test.txt`, score càng cao thì càng có khả năng là cùng người nói.
