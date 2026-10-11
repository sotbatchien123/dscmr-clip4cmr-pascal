# 05 - DCHMT

- Paper: Differentiable Cross-Modal Hashing via Multimodal Transformers, ACM MM 2022.
- Repo gốc: https://github.com/kalenforn/DCHMT .
- Repo kế nhiệm: https://github.com/kalenforn/clip-based-cross-modal-hash .
- Adapter ở đây dùng CLI của repo GỐC. Không dùng lẫn với YAML của repo kế nhiệm (tên cấu hình thường là DCMHT).

## Dữ liệu

MIRFlickr-25K: https://press.liacs.nl/mirflickr/mirdownload.html . Tải ảnh/tag và annotation. Theo README upstream, dùng các script `data/make_XXX.py` để tạo `caption.mat`, `index.mat`, `label.mat` trong `dataset/flickr25k` của repo.

Đường dẫn ảnh trong index cần trỏ đúng nơi giải nén. Chưa tải được upstream nên bước converter cụ thể và các file đã xử lý CHƯA được kiểm chứng; wrapper sẽ dừng khi thiếu chúng.

Trọng số: CLIP ViT-B/32, theo link trong repo OpenAI CLIP mà tác giả dẫn tới. Không đổi backbone khi đối chiếu kết quả.

```powershell
& $py 05_DCHMT\run.py --clip-weights weights\ViT-B-32.pt --bits 16 --batch-size 32 --dry-run
```

Bỏ dry-run khi đủ điều kiện. Các lựa chọn loss/hashing lấy từ ví dụ README; việc dùng cùng lựa chọn cho MIRFlickr còn phải đối chiếu paper/config theo revision. Batch 32 là điểm bắt đầu để thử tài nguyên, chưa phải cấu hình paper đã xác minh. Không báo kết quả này là reproduction chính xác khi chưa kiểm tra protocol.

## Điểm cần kiểm tra

- Số train/query/retrieval, nhãn, seed chia tập và độ dài mã.
- CLIP có bị freeze hay fine-tune theo đúng code/config không.
- mAP@ALL hoặc cutoff mà evaluator gốc thực sự dùng.
- README tác giả ghi một công thức vartheta trong hình bị sai và dẫn công thức (10) là đúng; phải đọc ghi chú này.
- Lưu checkpoint, full log và môi trường cho mỗi run. Code model chưa chạy trong phiên này.
