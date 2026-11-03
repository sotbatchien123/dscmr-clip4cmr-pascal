# DSCMR Assignment 3 — vòng 2: tuning hệ số + xác nhận 5 seed

## Tuning (3-fold CV trên 800 mẫu train, seed 42; không dùng test)

| Cấu hình | α / λ | β | τ | CV mAP trung bình (%) | Fold 1 / 2 / 3 |
|---|---:|---:|---:|---:|---|
| B0_original | 0.001 | 0.03 | 0.1 (–) | 68.78 ± 1.43 | 68.20 / 70.41 / 67.74 |
| B0_original | 0.001 | 0.1 | 0.1 (–) | 68.27 ± 1.14 | 67.97 / 69.53 / 67.31 |
| B0_original | 0.001 | 0.3 | 0.1 (–) | 68.32 ± 1.49 | 67.61 / 70.04 / 67.32 |
| B0_original **← chọn** | 0.01 | 0.03 | 0.1 (–) | 68.96 ± 1.61 | 68.44 / 70.77 / 67.68 |
| B0_original | 0.01 | 0.1 | 0.1 (–) | 68.30 ± 1.10 | 68.02 / 69.52 / 67.36 |
| B0_original | 0.01 | 0.3 | 0.1 (–) | 68.39 ± 1.38 | 67.83 / 69.96 / 67.37 |
| B0_original | 0.1 | 0.03 | 0.1 (–) | 68.79 ± 1.52 | 68.41 / 70.46 / 67.49 |
| B0_original | 0.1 | 0.1 | 0.1 (–) | 68.20 ± 1.28 | 67.52 / 69.68 / 67.41 |
| B0_original | 0.1 | 0.3 | 0.1 (–) | 68.47 ± 1.46 | 67.66 / 70.16 / 67.61 |
| B1_supcon | 0.03 | 0.03 | 0.05 | 66.57 ± 1.09 | 66.04 / 67.81 / 65.84 |
| B1_supcon | 0.03 | 0.03 | 0.1 | 68.39 ± 1.68 | 68.20 / 70.16 / 66.81 |
| B1_supcon **← chọn** | 0.03 | 0.03 | 0.2 | 68.80 ± 2.01 | 68.03 / 71.09 / 67.29 |
| B1_supcon | 0.1 | 0.03 | 0.05 | 66.41 ± 1.61 | 65.68 / 68.26 / 65.30 |
| B1_supcon | 0.1 | 0.03 | 0.1 | 67.96 ± 1.64 | 67.79 / 69.68 / 66.41 |
| B1_supcon | 0.1 | 0.03 | 0.2 | 68.43 ± 1.70 | 67.63 / 70.39 / 67.27 |
| B1_supcon | 0.3 | 0.03 | 0.05 | 66.60 ± 1.90 | 65.18 / 68.75 / 65.86 |
| B1_supcon | 0.3 | 0.03 | 0.1 | 67.75 ± 1.93 | 67.33 / 69.85 / 66.07 |
| B1_supcon | 0.3 | 0.03 | 0.2 | 67.97 ± 1.84 | 67.15 / 70.08 / 66.68 |

## Cấu hình xác nhận

- **B0_original***: hệ số (1, 0.01, 0.03), loss gốc, trọng số cố định
- **B1_supcon***: hệ số (1, 0.03, 0.03), SupCon τ=0.2, trọng số cố định
- **B2_adaptive***: hệ số (1, 0.01, 0.03), loss gốc, trọng số thích nghi
- **B3_combined***: hệ số (1, 0.03, 0.03), SupCon τ=0.2, trọng số thích nghi

## Kết quả test (720/80 split, 5 seed, % trung bình ± độ lệch chuẩn mẫu)

| Cấu hình | Chiều | mAP@ALL | R@1 | R@5 | R@10 |
|---|---|---:|---:|---:|---:|
| B0_original* | I2T | 66.02 ± 0.91 | 65.80 ± 3.11 | 84.40 ± 2.07 | 95.20 ± 0.84 |
| B0_original* | T2I | 68.42 ± 0.76 | 74.20 ± 2.28 | 93.40 ± 1.52 | 97.40 ± 0.89 |
| B1_supcon* | I2T | 67.17 ± 0.92 | 68.60 ± 2.41 | 85.80 ± 1.64 | 95.40 ± 0.89 |
| B1_supcon* | T2I | 68.24 ± 0.78 | 73.00 ± 2.55 | 93.00 ± 1.41 | 97.60 ± 0.55 |
| B2_adaptive* | I2T | 66.52 ± 0.86 | 66.40 ± 3.29 | 86.60 ± 1.14 | 95.20 ± 1.10 |
| B2_adaptive* | T2I | 68.40 ± 0.71 | 73.20 ± 1.48 | 92.60 ± 0.89 | 97.40 ± 0.55 |
| B3_combined* | I2T | 67.06 ± 0.64 | 68.20 ± 0.84 | 85.40 ± 3.21 | 94.60 ± 1.52 |
| B3_combined* | T2I | 68.07 ± 0.94 | 73.40 ± 2.30 | 93.60 ± 1.14 | 97.40 ± 0.89 |

## Chênh lệch mAP theo cặp seed so với B0_original* (điểm %)

| Cấu hình | Chiều | Seed 42 | Seed 43 | Seed 44 | Seed 45 | Seed 46 | Trung bình | Số seed tăng |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| B1_supcon* | I2T | +0.50 | +1.88 | +1.39 | +0.56 | +1.40 | +1.15 | 5/5 |
| B1_supcon* | T2I | -0.51 | +1.17 | -0.00 | -0.59 | -0.96 | -0.18 | 1/5 |
| B2_adaptive* | I2T | +0.18 | +0.41 | +0.24 | -0.17 | +1.85 | +0.50 | 4/5 |
| B2_adaptive* | T2I | -0.03 | -0.08 | +0.15 | +1.17 | -1.32 | -0.02 | 2/5 |
| B3_combined* | I2T | +0.86 | +1.59 | +0.08 | +0.27 | +2.40 | +1.04 | 5/5 |
| B3_combined* | T2I | +0.18 | +0.24 | +0.17 | -0.94 | -1.40 | -0.35 | 3/5 |

## Tỷ trọng từng thành phần loss (trung bình tới epoch tốt nhất, seed 42)

| Cấu hình | Loss nhãn | Không gian chung | Cặp ảnh–văn bản |
|---|---:|---:|---:|
| B0_original* | 86.8% | 2.8% | 10.4% |
| B1_supcon* | 65.0% | 27.1% | 7.9% |
| B2_adaptive* | 86.5% | 3.0% | 10.5% |
| B3_combined* | 65.4% | 26.6% | 8.0% |

## Phạm vi

- Tuning dùng ngân sách bằng nhau: 9 cấu hình × 3 fold cho B0 và cho B1; B2*/B3* không tuning thêm.
- CV score lấy epoch tốt nhất của mỗi fold nên hơi lạc quan, nhưng như nhau cho mọi cấu hình; chỉ dùng để xếp hạng.
- Tập test 100 cặp; chênh lệch dưới ~1 điểm có thể là nhiễu. 5 seed không phải 5 fold; không tuyên bố ý nghĩa thống kê.
- Tập test đã được xem ở các lần chạy trước; lần chạy này không dùng test để chọn cấu hình hay checkpoint.
- Trọng số thích nghi: quy tắc EMA tốc độ giảm loss có giới hạn, không phải GradNorm; tính mới chưa được chứng minh.
