# Bố cục dự án (cập nhật 01/11/2026)

| Thư mục / file | Nội dung | Có trên GitHub |
|---|---|---|
| `scripts/` | Huấn luyện, đánh giá, kiểm chứng, dựng báo cáo | Có |
| `common/`, `tests/` | Tính metric và kiểm tra metric | Có |
| `reports/` | Kết quả (JSON/CSV/log/hình) và báo cáo; `transition_20261014` là lịch sử chuyển đổi | Có (trừ checkpoint, đặc trưng, `history_before_20261014`) |
| `DSCMR/`, `CLIP4CMR/` | `source.json` ghi nguồn mã gốc; mã gốc tải vào `upstream/`, dữ liệu vào `data/`, lượt chạy cũ trong `runs/` | Chỉ file nhỏ (`source.json`, `status.json`, log, `data_ids/NOTE.md`) |
| `logs/` | Log các lượt chạy vòng 3 | Có |
| `checkpoints/` | Trọng số CLIP tải về | Không |
| `.venv310/`, `.feat_libs/` | Môi trường Python; open_clip + timm cài riêng cho vòng 3 | Không |
| `sources/` | Bài báo, đề bài (tài liệu có bản quyền) | Không |
| `GHI_NHO.md` | Ghi chú làm việc nội bộ | Không |

Chi tiết license: `LICENSE` (mã của dự án) và `THIRD_PARTY_NOTICES.md` (bên thứ ba).
