# Tài liệu dự án VLSP 2026 — SRED (Speaker Verification on Edge Devices)

Tài liệu này giải thích ba thứ:

1. **Cấu trúc thư mục**: mỗi file/folder dùng để làm gì.
2. **Các thực nghiệm đã làm**: làm gì, vì sao, kết quả ra sao.
3. **Sản phẩm đầu ra**: mỗi thực nghiệm sinh ra file gì, nằm ở đâu, dùng vào việc gì.

> Tài liệu kỹ thuật gốc (tiếng Anh): `README.md` (cài đặt, cách dùng) và `EXPERIMENTS.md` (nhật ký số liệu chi tiết).

---

## 1. Bài toán trong một đoạn

Cuộc thi **VLSP 2026 Vi-SRED26** yêu cầu xây hệ thống **xác minh người nói** (speaker verification): cho hai file
ghi âm, hệ thống trả về một **điểm số** cho biết hai file có cùng một người nói hay không. Điểm càng cao thì càng
có khả năng là cùng người.

- **Chỉ số đánh giá:** EER (Equal Error Rate, %): tỷ lệ lỗi tại ngưỡng mà số lần "nhận nhầm" bằng số lần "bỏ sót".
  **Càng thấp càng tốt.**
- **Ràng buộc:** mô hình dùng khi suy luận phải có **dưới 7 triệu tham số** (chạy được trên thiết bị biên).
- **Dữ liệu:**
  - `train`: 58.404 câu, 1.900 người nói, khoảng 144 giờ, 16 kHz mono.
  - `public_test`: 49.993 cặp thử (trial), 17.850 file, 902 người nói, có nhãn `target`/`nontarget`.
- **File nộp:** mỗi dòng `file1.wav file2.wav score`, đúng thứ tự với danh sách cặp thử.

---

## 2. Cài đặt và chạy nhanh

```bash
cd /home/jovyan/code/VLSP-2026/SRED
uv sync            # tạo .venv (torch 2.5.1 + CUDA 12.4)
source env.sh      # đặt biến môi trường: thư viện GPU, đường dẫn dữ liệu, cache torch hub
```

Dữ liệu nằm ngoài repo, tại `$SRED_DATA` (mặc định `/home/jovyan/data/VLSP2026/Vi-SRED26`).

Ba lệnh hay dùng nhất:

```bash
# 1) Đánh giá một mô hình có sẵn (chưa fine-tune) với mọi back-end chấm điểm
python -m sred.evaluate --model redimnet_M_mix --tag zeroshot

# 2) Fine-tune
python -m sred.train --base redimnet_M_mix --name F01 --train-from stage3 --bs 32 --crop-sec 2 --amp bf16

# 3) Tạo file nộp cho một tập test bất kỳ (ví dụ private test)
scripts/make_submission.sh <trials.txt> <audio_dir> private <out.txt> releases/version_141026_W01/sred_W01_redimnetM_141026.pt qmf_dur
```

---

## 3. Cấu trúc thư mục

```
VLSP-2026/
├── .git/                     # git repo (giữ ở thư mục gốc)
└── SRED/                     # toàn bộ dự án
    ├── DOCS.md               # tài liệu này
    ├── README.md             # hướng dẫn cài đặt + cách dùng (EN)
    ├── EXPERIMENTS.md        # nhật ký thực nghiệm, đầy đủ số liệu (EN)
    ├── env.sh                # biến môi trường, phải `source` trước khi chạy
    ├── pyproject.toml        # khai báo thư viện (quản lý bằng uv)
    ├── uv.lock               # khóa phiên bản thư viện
    ├── .venv/                # môi trường Python
    ├── .cache/torch/hub/     # mã nguồn ReDimNet v1/v2 tải từ torch hub
    │
    ├── sred/                 # MÃ NGUỒN CHÍNH (Python package)
    ├── scripts/              # script chạy thực nghiệm
    ├── third_party/wespeaker # mã nguồn WeSpeaker (để dựng ResNet34 / ECAPA)
    ├── pretrained/           # checkpoint huấn luyện sẵn (~432 MB)
    ├── lists/                # thông tin phụ trợ về dữ liệu (độ dài audio)
    │
    ├── exp/                  # MỌI ĐẦU RA CỦA THỰC NGHIỆM
    │   ├── ft/               #   checkpoint sau fine-tune
    │   ├── emb/              #   embedding đã trích xuất (cache)
    │   ├── logs/             #   log huấn luyện / đánh giá
    │   └── results.jsonl     #   bảng kết quả đánh giá (mỗi dòng một lần chạy)
    └── releases/             # các phiên bản đã nộp: version_<DDMMYY>_<hệ thống>/ (model, file điểm, zip, MANIFEST.json)
```

### 3.1. `sred/`: mã nguồn chính

Mỗi file đảm nhận một bước trong quy trình (xem mục 4):

| File | Vai trò (nói ngắn gọn) |
|---|---|
| `models.py` | "Kho mô hình" (`REGISTRY`): nạp ReDimNet v1/v2, WeSpeaker; đếm số tham số. ReDimNet2 chỉ có bản MLX/GGUF nên được chuyển sang PyTorch khi nạp. |
| `data.py` | Đọc danh sách train, danh sách cặp thử, nạp audio. |
| `augment.py` | Tăng cường dữ liệu ngay trên GPU: tiếng ồn đám đông (babble), tiếng ồn màu, vang phòng giả lập, cắt băng tần, méo micro, nén mu-law. Không cần bộ dữ liệu nhiễu ngoài. |
| `train.py` | Fine-tune: hàm mất mát sub-center AAM-softmax, tuỳ chọn chưng cất tri thức (KD) từ mô hình thầy, đóng băng một phần mạng, bf16, lưu checkpoint theo bước, lấy trung bình trọng số. |
| `embed.py` | Trích xuất embedding (vector đặc trưng giọng nói) cho từng file, lưu cache vào `exp/emb/`. |
| `scoring.py` | Chấm điểm cặp: cosine, mean-centering, AS-norm. |
| `backend.py` | Các phép biến đổi tuyến tính học trên embedding train: LDA, WCCN, whitening, PLDA. |
| `calibrate.py` | **QMF**: hiệu chỉnh điểm theo độ dài audio (giải thích ở mục 5). |
| `metrics.py` | Tính EER và minDCF. |
| `evaluate.py` | Gộp mọi bước: trích embedding → chấm điểm bằng mọi back-end → ghi kết quả vào `exp/results.jsonl` → (tuỳ chọn) ghi file nộp. |
| `protocol.py` | **Giao thức chống overfit**: chia public test thành P-dev / P-hold và kiểm định thống kê bằng bootstrap (mục 6.5). |

### 3.2. `scripts/`: script chạy thực nghiệm

| Script | Dùng cho |
|---|---|
| `run_zeroshot.sh` | Đánh giá các mô hình có sẵn, chưa fine-tune (E00–E03). |
| `quick_subset_eval.py` | Đánh giá nhanh trên 1.500 cặp ngẫu nhiên cố định (sàng lọc mô hình). |
| `f01.sh`, `f03.sh`, `f04.sh`, `f05.sh` | Chạy từng thí nghiệm fine-tune cùng tên. |
| `k01.sh` | Fine-tune có chưng cất tri thức từ ReDimNet2-B6 (K01). |
| `pipeline_v2.sh` | Chạy các thí nghiệm "đăng ký trước" của Protocol v2; chạy lại được sau khi container khởi động lại (tự resume). |
| `queue1.sh`, `queue2.sh`, `queue3.sh` | Hàng đợi: chờ một job xong rồi chạy tiếp đánh giá. |
| `make_submission.sh` | Tạo file nộp cho một tập test bất kỳ và tự kiểm tra định dạng (đúng thứ tự cặp, điểm hợp lệ, đủ số dòng). |

### 3.3. `pretrained/`: mô hình huấn luyện sẵn

Tải từ HuggingFace (máy này bị chặn tải từ GitHub release).

| Thư mục | Nội dung |
|---|---|
| `redimnet_v1/` | ReDimNet v1: các bản S, M, b3, b4, b5. **`M-vb2+vox2+cnc-ft_mix.pt` là mô hình chính.** |
| `causal__redimnet2-b4-vox2-lm-mlx/`, `...b3...` | ReDimNet2 dạng MLX (chuyển sang PyTorch khi nạp). |
| `OpenASR__redimnet2-b6-cn/` | ReDimNet2-B6 (12,5M tham số): chỉ dùng làm **mô hình thầy**, quá giới hạn 7M nên không dùng để nộp. |
| `Wespeaker__*`, `gaunernst__*` | Các mô hình WeSpeaker (ResNet34, ECAPA, SimAM-ResNet34). |
| `official/` | Để dành cho checkpoint chính thức của ban tổ chức (hiện trống). |

> Lưu ý: chỉ được nộp bài bằng checkpoint nằm trong **danh sách được ban tổ chức phê duyệt**.

### 3.4. `lists/`

| File | Nội dung |
|---|---|
| `train_lengths.json` | Độ dài từng file train (dùng khi cắt đoạn ngẫu nhiên lúc huấn luyện). |
| `public_durations.npy` | Độ dài từng file public test (dùng cho QMF và phân tích lỗi theo độ dài). |

---

## 4. Quy trình tổng thể

```
 file audio ──► mô hình (ReDimNet-M, 4,81M tham số) ──► embedding (vector 192 chiều)
                                                              │
                                     ┌────────────────────────┘
                                     ▼
              chấm điểm cặp (cosine, có thể trừ trung bình / chuẩn hoá)
                                     │
                                     ▼
              hiệu chỉnh QMF theo độ dài 2 file  ──►  điểm cuối  ──►  file nộp
```

Nói đơn giản: mô hình biến mỗi đoạn ghi âm thành một "dấu vân tay giọng nói" (embedding). Hai dấu vân tay càng giống
nhau (cosine càng cao) thì càng có khả năng cùng người. QMF chỉnh lại điểm này, vì cặp có audio ngắn thường cho điểm
kém tin cậy hơn.

---

## 5. Giải thích nhanh các thuật ngữ

| Thuật ngữ | Ý nghĩa |
|---|---|
| **EER** | Tỷ lệ lỗi cân bằng (%). Chỉ số chính, càng thấp càng tốt. |
| **minDCF** | Chi phí phát hiện nhỏ nhất; nhạy với lỗi "nhận nhầm" ở ngưỡng chặt. Càng thấp càng tốt. |
| **Zero-shot** | Dùng mô hình có sẵn, không huấn luyện thêm. |
| **Fine-tune** | Huấn luyện tiếp mô hình có sẵn trên dữ liệu train tiếng Việt. |
| **Back-end** | Bước xử lý sau embedding để ra điểm: cosine, centering, AS-norm, LDA, PLDA... |
| **Mean-centering** | Trừ đi embedding trung bình của dữ liệu train trước khi tính cosine. |
| **QMF** | *Quality Measure Function*: hồi quy logistic trên [điểm, log độ dài ngắn nhất, log độ dài dài nhất, và tích của chúng]. Được học trên ~320 nghìn cặp **tạo từ tập train** (không dùng nhãn test). |
| **KD / RKD** | Chưng cất tri thức: mô hình nhỏ (trò) học bắt chước mô hình lớn (thầy, B6). RKD = bắt chước *quan hệ* giữa các embedding. |
| **WiSE-FT** | Trộn trọng số: `0,5 × mô hình fine-tune + 0,5 × mô hình gốc`. Giữ được cái lợi của fine-tune mà ít "trôi" khỏi mô hình gốc. |
| **Freeze BN** | Đóng băng thống kê BatchNorm khi fine-tune. |
| **sub1500 / dev2000** | Tập con cố định 1.500 / 2.000 cặp của public, dùng để sàng lọc nhanh / chọn checkpoint (chỉ ở giai đoạn đầu). |
| **P-dev / P-hold** | Hai nửa của public test, chia theo người nói (mục 6.5). P-dev để ra quyết định, P-hold chỉ mở **một lần** ở cuối. |
| **Bootstrap, P(better)** | Lấy mẫu lại theo người nói 200 lần để ước lượng độ tin cậy. P(B better) > 0,95 mới coi B thật sự tốt hơn A. |

---

## 6. Các thực nghiệm đã làm

Tên thí nghiệm theo quy ước: **E** = zero-shot, **T** = mô hình thầy, **F** = fine-tune, **K** = fine-tune có KD,
**W** = trộn trọng số (WiSE-FT).

### 6.1. Giai đoạn 1: thử các mô hình có sẵn (zero-shot)

**Mục tiêu:** chọn mô hình nền tốt nhất trong danh sách được phép.

| ID | Mô hình | Tham số | EER sub1500 (cosine) | EER public (cosine) |
|---|---|---|---|---|
| **E00** | ReDimNet-M mix | 4,81M | 6,68 | **5,87** |
| E03 | ReDimNet-S mix | 3,15M | 7,48 | – |
| E02 | WeSpeaker ResNet34-LM | 6,63M | 10,01 | – |
| E01 | ReDimNet2-B4 vox2 | 6,66M | 10,40 | – |
| T00 | ReDimNet2-B6 (thầy, quá 7M) | 12,46M | 5,53 | – |

**Kết luận:** ReDimNet-M tốt nhất trong nhóm dưới 7M → chọn làm **mô hình chính**. B6 mạnh hơn nhưng quá lớn → chỉ làm thầy.

### 6.2. Giai đoạn 2: so sánh các back-end trên E00

| Back-end | EER public |
|---|---|
| Cosine | 5,87 |
| Mean-centering | 5,50 |
| AS-norm (300) | 5,65 |
| LDA-128 | 5,35 |
| WCCN | 5,51 |
| Whitening | 5,94 |
| PLDA | 6,39 |
| **Centering + QMF (độ dài)** | **5,00** |

**Kết luận:** QMF theo độ dài cho mức giảm lớn nhất. Các biến thể thêm đặc trưng AS-norm hoặc độ lớn embedding
vào QMF đều làm **tệ hơn** (kết quả 4,97 trước đó là do lỗi rò rỉ dữ liệu, đã sửa).

**Phân tích lỗi theo độ dài:** lỗi tập trung ở cặp có file **ngắn dưới 2 giây** (EER ~10,7%), trong khi cặp từ 8 giây
trở lên chỉ ~1,3%. Đây là lý do QMF theo độ dài hiệu quả.

### 6.3. Giai đoạn 3: fine-tune ReDimNet-M

**Công thức chung:** sub-center AAM-softmax, tăng cường dữ liệu trên GPU, batch 32, bf16, AdamW, đóng băng các lớp đầu
(stem → stage2) vì GPU dùng chung nên huấn luyện toàn bộ quá chậm.

| ID | Thay đổi | Kết quả | Kết luận |
|---|---|---|---|
| F01 | AAM + augmentation, crop 2 s, 3.000 bước | public cosine 5,51 · QMF 5,09 | Tốt hơn với cặp ngắn, kém hơn với cặp dài. |
| F02 | F01 + RKD từ B6 | dev2000 không cải thiện, chậm hơn 1,6 lần | **Dừng sớm.** |
| **F03** | F01 + **freeze BN** + crop ngẫu nhiên 1,5–3 s + lr 2e-5 | public cosine 5,31 · **QMF 4,88** | Tốt nhất giai đoạn này. |
| F04 | F03 + mở thêm lớp (từ stage1), 4.000 bước | – | **Tạm dừng** ngay đầu, không có kết quả. |

### 6.4. Kiểm định thống kê (11/10/2026)

Bootstrap theo người nói (200 lần) trên public:

- **QMF chắc chắn có ích:** cải thiện so với cosine có khoảng tin cậy không chứa 0.
- **Lợi ích của fine-tune nhỏ và chưa chắc chắn:** F03 so với E00 (cùng QMF) chỉ −0,28 EER, P = 0,94 (< 0,95).
- **Khác biệt giữa người nói rất lớn:** hai nửa người nói khác nhau cho EER 3,3% và 5,5%. Nghĩa là điểm trên private
  có thể lệch hơn 1 điểm so với public, bất kể hệ thống nào.
- **Rủi ro overfit:** dev2000 là tập con của public, và việc chọn back-end/công thức được làm sau khi xem kết quả public.

→ Dẫn tới việc xây **Protocol v2**.

### 6.5. Giai đoạn 4: Protocol v2, giao thức chống overfit (12/10/2026)

**Ý tưởng:** chia public test theo người nói (dùng hash md5, cố định) thành hai nửa không giao nhau:

- **P-dev** (228 người, 21.085 cặp): dùng cho **mọi quyết định**.
- **P-hold** (236 người, 16.394 cặp): **chỉ mở một lần** ở cuối, để ước lượng trung thực.

**Quy tắc:** chỉ chấp nhận hệ thống B thay cho A nếu P(B tốt hơn) > 0,95 trên P-dev. Back-end cố định là QMF.
Không chọn checkpoint theo public nữa; hệ thống cuối = **trung bình trọng số các checkpoint cuối** (`avg.pt`).
Các thí nghiệm được **đăng ký trước** khi chạy.

Mốc so sánh trên P-dev (cosine / QMF): E00 5,78 / 5,02 · F01 5,66 / 5,19 · F03 5,08 / **4,60**.

| ID | Mô tả | P-dev (cosine / QMF) | So sánh | Quyết định |
|---|---|---|---|---|
| T00 | B6 zero-shot (thầy) | 4,85 / 4,07 | Tốt hơn F03 ~0,5 → đủ điều kiện thử KD | Chỉ làm thầy |
| F05 | Công thức F03, **9.000 bước** (gấp 3), avg 3 checkpoint cuối | 5,47 / 5,01 | Tệ hơn F03 (+0,42) | **Loại**: huấn luyện lâu làm mô hình "trôi" khỏi bản gốc |
| **W01** | **WiSE-FT: 0,5·F03 + 0,5·M gốc** (không huấn luyện thêm) | 5,08 / 4,55 | Hoà F03; **vượt E00 với P = 0,98** | **Chọn làm hệ thống cuối** |
| K01 | F05 + RKD từ B6 (w = 10) | 5,52 / 4,51 | So với F03: P = 0,73; minDCF kém hơn | Không chấp nhận |

> F05 bị chạy lại một lần vì container khởi động lại giữa chừng (bản chết lưu ở `F05_M_s3_bnfz_long.dead`).

**Quyết định (đưa ra trước khi mở P-hold):** hệ thống cuối = **W01 + QMF(độ dài)**.

### 6.6. Mở P-hold (một lần duy nhất)

| Hệ thống | Cosine | QMF | minDCF | QMF so với E00 |
|---|---|---|---|---|
| E00 (ReDimNet-M gốc) | 5,81 | 5,32 | 0,210 | – |
| F03 | 5,47 | 5,17 | 0,156 | −0,15 (P = 0,80) |
| **W01 (cuối)** | 5,42 | 5,15 | 0,161 | −0,17 (P = 0,94) |
| K01 | 5,55 | 5,05 | 0,166 | −0,27 (P = 0,93) |

**Đọc kết quả thế nào:**

- Cải thiện **có thật về chiều hướng** nhưng **nhỏ hơn** so với P-dev (W01: −0,47 trên P-dev → −0,17 trên P-hold).
  Tức là con số trên P-dev hơi lạc quan.
- **minDCF cải thiện rõ** (0,210 → 0,161).
- **Kỳ vọng thực tế trên private:** tốt hơn mô hình gốc + QMF khoảng **0,2–0,5 EER**, minDCF tốt hơn rõ rệt.

---

## 7. Sản phẩm đầu ra

### 7.1. Bảng tra: thực nghiệm → file đầu ra

| Thực nghiệm | Checkpoint (`exp/ft/`) | Embedding (`exp/emb/`) | Log (`exp/logs/`) | File nộp (`releases/`) |
|---|---|---|---|---|
| E00 | (dùng `pretrained/redimnet_v1/M-...mix.pt`) | `redimnet_M_mix/` | `zeroshot_redimnet_M_mix.log` | `version_111026_E00/sred_E00_public_111026.txt` |
| E01–E03 | – | – | `quick_*.log` | – |
| T00 | (dùng `pretrained/OpenASR__redimnet2-b6-cn`) | `redimnet2_b6_mix_teacher/` | `quick_b6.log`, `eval_T00_B6.log` | – |
| F01 | `F01_M_s3_aug/` (`best.pt`, `last.pt`) | `F01_M_s3_aug_best/` | `F01.log`, `eval_F01.log` | `version_111026_F01/sred_F01_public_111026.txt` |
| F02 | `F02_M_s3_aug_rkdB6/` | – | `F02.log` | – |
| F03 | `F03_M_s3_aug_bnfz_var/` (`best.pt`, `last.pt`) | `F03_M_s3_aug_bnfz_var_best/` | `F03.log`, `eval_F03.log` | `version_111026_F03/sred_F03_public_111026.txt` |
| F04 | `F04_M_s1_aug_bnfz_var/` (chỉ log + args) | – | `F04.log` | – |
| F05 | `F05_M_s3_bnfz_long/` (`step1500…9000.pt`, `avg.pt`) | `F05_M_s3_bnfz_long_avg/` | `F05.log`, `protocol_F05.log` | – |
| K01 | `K01_M_s3_bnfz_long_rkdB6/` (`step*.pt`, `avg.pt`) | `K01_M_s3_bnfz_long_rkdB6_avg/` | `K01.log`, `protocol_K01.log` | – |
| **W01** | **`W01_wise_F03_M/avg.pt`** | `W01_wise_F03_M_avg/` | `protocol_W01.log` | **`version_141026_W01/sred_W01_public_141026.{txt,zip}`** |
| Mở P-hold | – | – | `protocol_reveal.log` | – |

### 7.2. Ý nghĩa từng loại đầu ra

**Checkpoint (`exp/ft/<tên>/`)**

| File | Ý nghĩa |
|---|---|
| `args.json` | Toàn bộ siêu tham số của lần chạy (để tái lập). |
| `log.txt` | Loss, độ chính xác, EER kiểm tra theo bước. |
| `best.pt` / `last.pt` | Checkpoint tốt nhất theo dev2000 / checkpoint cuối (chỉ ở F01–F03). |
| `step1500.pt` … `step9000.pt` | Checkpoint lưu mỗi 1.500 bước (Protocol v2). |
| `avg.pt` | Trung bình trọng số 3 checkpoint cuối: **đây là "hệ thống" trong Protocol v2**. |
| `resume.pt` | Trạng thái để tiếp tục huấn luyện nếu bị ngắt. |

Mỗi checkpoint ReDimNet-M khoảng 19 MB; một thư mục F05/K01 khoảng 236 MB.

**Embedding (`exp/emb/<hệ thống>/`)**: hai file `.npy` cho mỗi hệ thống:

- `public_modefull_<hash>.npy`: embedding của 17.850 file public test.
- `traincohort5_modefull_<hash>.npy`: embedding của 5 câu/người từ train (~9,5 nghìn câu), dùng cho centering, AS-norm, QMF.

`<hash>` được tính từ danh sách file và tuỳ chọn trích xuất, nên đổi dữ liệu thì cache tự sinh file mới.
Có cache thì đánh giá lại chỉ mất vài giây thay vì ~25 phút.

**Bảng kết quả (`exp/results.jsonl`)**: mỗi dòng là một lần chạy `sred.evaluate`, gồm thời gian, mô hình, số tham số,
thời gian trích embedding, và EER + minDCF của **mọi back-end** (cosine, center, asnorm, lda, plda, qmf_dur…).

**File nộp (`releases/`)**: 49.993 dòng, đúng định dạng `file1.wav file2.wav score`:

| File | Hệ thống | Ghi chú |
|---|---|---|
| **`version_141026_W01/sred_W01_public_141026.{txt,zip}`** | **W01 + QMF** | **Bản nộp chính thức** (4,81M tham số) |
| `version_111026_F03/sred_F03_public_111026.txt` | F03 + QMF | Dự phòng |
| `version_111026_E00/sred_E00_public_111026.txt` | ReDimNet-M gốc + centering + QMF | Mốc so sánh |
| `version_111026_F01/sred_F01_public_111026.txt` | F01, cosine | Bản cũ |

### 7.3. Tạo file nộp cho private test

```bash
cd /home/jovyan/code/VLSP-2026/SRED
scripts/make_submission.sh <private_trials.txt> <private_audio_dir> private releases/<version>_W01/sred_W01_private_<DDMMYY>.txt \
    releases/version_141026_W01/sred_W01_redimnetM_141026.pt qmf_dur
```

Script tự in ba dòng kiểm tra: `mismatched lines: 0`, `bad score lines: 0`, và số dòng hai file phải bằng nhau.

---

## 8. Tóm tắt bài học

1. **Hiệu chỉnh điểm theo độ dài (QMF) là cải tiến chắc chắn nhất**: rẻ, không tốn tham số, ổn định trên cả hai nửa dữ liệu.
2. **Fine-tune ngắn + đóng băng BN** tốt hơn fine-tune dài. Huấn luyện lâu làm mô hình "quên" kiến thức gốc, nhất là với audio dài.
3. **Trộn trọng số (WiSE-FT)** giữ được lợi ích fine-tune mà an toàn hơn, nên được chọn làm bản cuối.
4. **KD từ B6** chỉ bù lại phần mô hình bị trôi khi huấn luyện dài, chưa tạo ra lợi ích rõ ràng.
5. **Audio ngắn (< 2 s) vẫn là điểm yếu lớn nhất** (EER ~10–11%). Đây là hướng cải thiện tiếp theo.
6. **Phải có giao thức đánh giá nghiêm ngặt**: con số trên tập dùng để ra quyết định luôn lạc quan hơn thực tế.

## 9. Lưu ý vận hành

- GPU L4 **dùng chung** với container khác (~15,5 GB đã bị chiếm, 100% tải): suy luận ~80 lần thời gian thực, huấn luyện chậm.
- Đường dẫn trong thư mục public test có chứa ID người nói. Thông tin này **chỉ** dùng để chia P-dev/P-hold và làm bootstrap,
  **không** hệ thống nào dùng nó để chấm điểm.
- Dự án đã được chuyển vào thư mục `SRED/`; các script và `env.sh` đã cập nhật đường dẫn `/home/jovyan/code/VLSP-2026/SRED`.
