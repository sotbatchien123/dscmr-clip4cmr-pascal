# Khôi phục ID ảnh cho đặc trưng DSCMR: THẤT BẠI (30/10/2026)

- `scripts/analysis/dscmr_ids_torchvision.py`: VGG19 torchvision (fc7 sau ReLU, 2 kiểu tiền xử lý) so với đặc trưng của tác giả cho cos ghép trung bình 0.19 → không cùng mô hình hoặc trọng số (có thể VGG19 Caffe hoặc đã fine-tune). Không thể ghép trực tiếp.
- `scripts/analysis/dscmr_ids_matching.py`: ghép theo cấu trúc tương đồng trong từng lớp (QAP), rồi kiểm chứng chéo bằng ridge + Hungarian. Chỉ 3% cặp tái tạo được → các cặp ghép không đáng tin.
- Kết luận: không có ID cho split 800/100 của DSCMR. Không dùng bất kỳ file ghép nào. Thí nghiệm "DSCMR + đặc trưng CLIP" phải chạy trên split 800/200 của tác giả CLIP4CMR, tức là khác tập test với DSCMR gốc.

## Lần thử 2 (31/10/2026): VGG19 Caffe gốc (Oxford) — vẫn THẤT BẠI

- Trọng số: bản Keras port từ VGG19 Caffe của Oxford (`fchollet/deep-learning-models` v0.1, sha256 `c872d82e…`). Không đưa vào repo: đặt ở `checkpoints/vgg19_caffe/vgg19_keras_caffe.h5` hoặc trỏ biến môi trường `VGG19_CAFFE_H5` tới file. Kiểm tra: ảnh `2008_000716` → lớp 404 "airliner", xác suất 0.51.
- Script: `scripts/analysis/dscmr_ids_caffe.py`. Tiền xử lý kiểu Caffe: BGR, trừ mean. Thử 3 kiểu cắt ảnh: warp224, center256, tencrop256. Đặc trưng lưu ở `caffe_vgg19_fc7.npz`.
- Cosine tốt nhất giữa mỗi dòng DSCMR và ảnh: trung vị 0.27–0.31; chỉ 14–58 ảnh khác nhau được chọn làm best match (hubness).
- Mức kích hoạt trung bình theo từng nơ-ron: tương quan Spearman ≈ 0.00 → chiều của đặc trưng DSCMR **không** tương ứng với nơ-ron fc7 của VGG19 gốc (có thể đã fine-tune mạnh, đã xáo chiều, hoặc dùng mạng khác).
- Đặc trưng văn bản không âm (đầu ra CNN văn bản đã train) → không tái tạo được.
- Issue #4 của repo DSCMR: split có thể theo Corr-AE (Feng et al. 2014). Các bài liên quan đều ghi "chia ngẫu nhiên 800/100/100, đều theo lớp"; không bài nào công khai danh sách ảnh.
- Kết luận: không tái tạo được ánh xạ, kể cả một phần đáng tin. Đã xóa file ghép (`recovered_ids_caffe.json`) vì không hợp lệ.
