# Kết quả DSCMR và CLIP4CMR: 5.000 + 1.000 epoch

Đã hoàn tất đúng cấu hình yêu cầu cho cả hai mô hình. Giai đoạn đầu ghi nhận checkpoint tốt nhất trên validation trong 5.000 epoch. Sau đó khởi tạo lại và huấn luyện đúng 1.000 epoch trên toàn bộ 800 mẫu train; đánh giá checkpoint epoch 1.000 trên test. Best validation epoch không quyết định thời lượng refit trong cấu hình này.

| Mô hình | Chiều | mAP (%) | R@1 (%) | R@5 (%) | R@10 (%) | mAP lượt trước (%) | Test |
|---|---|---:|---:|---:|---:|---:|---:|
| DSCMR | I2T | 63.3575 | 64.0 | 82.0 | 94.0 | 64.6654 | 100 |
| DSCMR | T2I | 65.8603 | 68.0 | 93.0 | 96.0 | 69.7477 | 100 |
| CLIP4CMR | I2T | 65.0305 | 72.5 | 82.0 | 90.5 | 70.9177 | 200 |
| CLIP4CMR | T2I | 65.9036 | 70.0 | 87.0 | 94.0 | 72.0938 | 200 |

- DSCMR: đủ 5.000 epoch validation + 1.000 epoch refit = 6.000 epoch qua hai mô hình khởi tạo riêng. Best validation epoch: 61. Thời gian runner: 1158.6 giây.
- CLIP4CMR: đủ 5.000 epoch validation + 1.000 epoch refit = 6.000 epoch qua hai mô hình khởi tạo riêng. Best validation epoch: 69. Thời gian runner: 273.3 giây.

Mỗi giai đoạn validation dùng 720 train/80 validation tách từ 800 mẫu train. Không dùng test để chọn checkpoint trong runner. Test của hai lượt cũ đã từng được báo cáo; đây là thí nghiệm tiếp theo trên cùng test, không phải một tập kiểm định mới chưa từng xem.

DSCMR: đặc trưng tác giả, test100. CLIP4CMR: đặc trưng CLIP RN50 đóng băng tự tạo từ UIUC, split800/200 seed42; ghép caption theo thứ tự nguồn và cắt77token. Không thay đổi dữ liệu, seed, loss, learning rate hoặc batch so với lượt trước.

Có thể đối chiếu lượt mới/lượt trước trong từng mô hình. Không xếp hạng trực tiếp DSCMR với CLIP4CMR vì test khác nhau; chưa xác minh tái lập đúng split/tiền xử lý của survey. Thêm epoch không bảo đảm tăng mAP.

Đã kiểm tra đủ số epoch trong history.json và refit_history.json, tính lại metrics từ test_features.npz và đối chiếu mAP hai chiều với evaluator tác giả.

Kết quả và checkpoint cũ vẫn được giữ. Kết quả mới:
- DSCMR/runs/pascal_selection5000_refit1000_seed42/
- CLIP4CMR/runs/pascal_selection5000_refit1000_seed42/

Trong mỗi thư mục: result.json, protocol.json, history.json, refit_history.json, validation_best.pt, model.pt, test_features.npz, verification.json. model.pt là checkpoint refit epoch1.000; validation_best.pt thuộc mạng giai đoạn720train.

Chỉ số mAP và R@K trong bảng là %. R@K là tỷ lệ truy vấn có ít nhất một kết quả cùng lớp ở top K.
