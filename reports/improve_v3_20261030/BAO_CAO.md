# Vòng 3 — Mục tiêu +6 điểm mAP cho CLIP4CMR và DSCMR (2026-10-31)

## Giao thức

- **Dữ liệu:** split tác giả CLIP4CMR (Pascal Sentence, 800 train / 200 test), khôi phục từ bản sao HF (bên thứ ba), xem `reports/clip4cmr_author_split_20261030/`.
- **Chọn cấu hình:** 3-fold CV phân tầng, chỉ trên 800 dòng train. Code: `scripts/improve_v3.py --stage cv` và `scripts/finetune_v3.py --stage cv`.
- **Xác nhận:**
  - 5 seed (42–46), split cố định 720/80, checkpoint chọn theo validation.
  - Test chỉ mở **một lần**, sau khi train xong tất cả seed.
- **Chỉ số:** mAP@ALL, liên quan theo nhãn. Kết quả là trung bình ± độ lệch chuẩn của 5 seed.
- **Không dùng re-ranking:** CV cho thấy nó chỉ thêm tối đa 0.15 điểm và cần thông tin transductive.

## Kết quả CV (TB 2 chiều, %)

| Cấu hình | CV |
|---|---|
| RN50 c4_base (baseline) | 66.60 |
| DFN5B đóng băng, A+B | 71.57 |
| Ensemble DFN5B+EVA02+SigLIP, đầu DSCMR (ds_capmean) | 73.03 |
| Fine-tune DFN5B 2 block cuối + A+B (`c4_AB_L2`) | **73.24** |
| Fine-tune 4 block cuối + A+B | 72.90 |
| Fine-tune 2 block cuối + đầu DSCMR | 72.71 |
| Fine-tune 4 block cuối + đầu DSCMR (`ds_L4`) | **73.04** |

Cấu hình chính chốt **trước khi mở test**:
- CLIP4CMR: `c4_AB_L2`.
- DSCMR: `ds_L4`. Ensemble ds_capmean (CV 73.03, gần như bằng) là cấu hình phụ.

## Kết quả test (5 seed)

| Cấu hình | I2T | T2I | TB | Δ so với baseline cùng cặp seed |
|---|---|---|---|---|
| RN50 c4_base (baseline CLIP4CMR) | 68.48 ± 0.63 | 67.87 ± 0.40 | 68.17 | — |
| RN50 ds_capmean (đầu DSCMR trên RN50) | 68.86 ± 0.47 | 69.89 ± 0.28 | 69.38 | — |
| DFN5B đóng băng + A+B | 73.71 ± 0.16 | 73.53 ± 0.20 | 73.62 | +5.45 so với c4_base |
| **Fine-tune `c4_AB_L2` (chính, CLIP4CMR)** | **76.63 ± 0.39** | **75.68 ± 0.46** | **76.15** | **+7.98** so với c4_base, 5/5 seed |
| **Fine-tune `ds_L4` (chính, DSCMR)** | 74.78 ± 0.51 | 72.70 ± 0.62 | 73.74 | +4.36 so với RN50 ds; +5.69 so với B0 gốc 68.05 |
| Ensemble ds_capmean (phụ, DSCMR) | 73.75 ± 0.33 | 74.38 ± 0.47 | 74.06 | +4.68 so với RN50 ds; +6.01 so với B0 gốc |

Kết quả từng seed (TB 2 chiều):

| Cấu hình | Seed 42–46 |
|---|---|
| c4_AB_L2 | 76.57 / 75.91 / 76.43 / 76.40 / 75.47 |
| c4_base | 68.36 / 67.47 / 68.17 / 68.00 / 68.88 |
| ds_L4 | 74.33 / 73.62 / 73.26 / 73.79 / 73.71 |
| RN50 ds | 69.89 / 69.31 / 69.51 / 69.35 / 68.81 |

## Kết luận

- **CLIP4CMR: ĐẠT mục tiêu.**
  - Tăng +7.98 điểm (68.17 → 76.15), khoảng +11.7% tương đối.
  - Cả 5 seed và cả hai chiều đều tăng.
  - Đóng góp: encoder mạnh hơn (đóng băng) +5.45; fine-tune 2 block cuối thêm khoảng +2.5.
- **DSCMR: CHƯA ĐẠT với cấu hình chính.**
  - `ds_L4` đạt +5.69 so với B0 gốc (68.05), +4.36 so với baseline cùng split.
  - Ensemble phụ đạt +6.01 so với B0. Tuy nhiên, nó chỉ là cấu hình phụ, và mức vượt ngưỡng (0.01) nằm trong nhiễu. **Không được trình bày nó là cấu hình chính** chỉ vì đã thấy kết quả test.
  - So với B0, đây không phải so sánh cùng cặp: B0 dùng split và test 100 mẫu của DSCMR, không khôi phục được ID (xem `DSCMR/data_ids/NOTE.md`).
  - So sánh công bằng là với RN50 ds_capmean trên cùng split: +4.4 đến +4.7 điểm.

## Hạn chế

- Split tác giả lấy từ bản sao bên thứ ba. Đã kiểm chứng: 922/1000 ảnh và 938/1000 văn bản khớp.
- DSCMR "cải tiến" ở đây thực chất là đầu DSCMR trên đặc trưng CLIP-họ. Không còn là VGG19 + Word2Vec như bài gốc.
- Fine-tune chỉ áp dụng cho nhánh ảnh; nhánh văn bản đóng băng.

## File

- **Code:** `scripts/improve_v3.py`, `scripts/finetune_v3.py`, `scripts/extract_encoders.py`.
- **CV:** `cv/`, `finetune/cv/`.
- **Test:**
  - `confirm_results.json` (đóng băng).
  - `finetune/confirm/<cfg>/test_results.json`.
  - Đặc trưng test của từng seed: `test_features.npz`.
- **Log:** `logs/v3_confirm_*.log`.
