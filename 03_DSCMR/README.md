# 03 - DSCMR

- Code tác giả: https://github.com/penghu-cs/DSCMR
- Paper: Deep Supervised Cross-Modal Retrieval, CVPR 2019.
- Pascal Sentence: https://vision.cs.uiuc.edu/pascal-sentences/

`train.py` import model, loss và evaluator từ code tác giả, bổ sung CLI, seed, lưu checkpoint/embedding/history. Chưa có upstream cục bộ, feature hay PyTorch; driver chưa chạy end-to-end.

## Đầu vào chính xác

Thư mục feature phải có 6 file: `train_img.mat`, `test_img.mat`, `train_txt.mat`, `test_txt.mat`, `train_img_lab.mat`, `test_img_lab.mat`; key trong từng MAT trùng tên file không có đuôi.

```powershell
& $py scripts\validate_data.py dscmr --root datasets\pascal_features
```

Ảnh/caption gốc không phải các feature này. Phải xác minh encoder, phiên bản feature, số chiều, cách xử lý caption, nhãn và split với paper. Không thay VGG/Sentence-CNN bằng CLIP rồi gọi là tái lập DSCMR nguyên bản.

## Hai chế độ

1. `upstream-test`: dùng vòng lặp tác giả, chọn model có mAP cao nhất trên tập được repo đặt tên `test`. Dùng để đối chiếu đúng hành vi upstream; cần nêu rõ test đã tham gia chọn checkpoint.
2. `fixed-last` (mặc định): không đánh giá test trong training, đánh giá sau epoch cuối. Giữ model/loss nhưng đổi quy trình chọn checkpoint; không gộp kết quả với chế độ 1.

```powershell
& $py 03_DSCMR\train.py --upstream upstream\dscmr --data datasets\pascal_features --out 03_DSCMR\results\fixed_last_seed42 --selection fixed-last --device cpu --epochs 500
```

Với chế độ upstream dùng CUDA, đổi thành `--selection upstream-test --device cuda` và thư mục output mới. 500 epoch, batch 100, Adam lr 1e-4, alpha 1e-3, beta 1e-1 được lấy làm cấu hình xuất phát từ main.py, chưa được xác minh khớp tất cả thiết lập bài gốc.

Driver dùng keyword `num_epochs` để tránh lỗi lệch vị trí tham số giữa main.py và train_model.py. Cần test thực sau khi clone vì revision upstream có thể thay đổi.

## Kết quả

Hai mAP, history, checkpoint, embedding; ghi đầy đủ chế độ chọn model. Bảng tham chiếu của survey lấy từ nguồn khác, vì vậy phải xác minh lại dòng của bài gốc trước khi điền chênh lệch.
