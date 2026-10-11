# Phân công và bằng chứng đóng góp

Đây là PHÂN CÔNG DỰ KIẾN cho 5 thành viên, chưa phải lịch sử đóng góp.

| Vị trí | Phương pháp | Công việc cần hoàn thành | Bằng chứng |
|---|---|---|---|
| Thành viên 1 | CLIP | Dataset manifest, embedding, evaluation; tổng hợp bảng nhóm | Commit thật + run logs |
| Thành viên 2 | ViLT | Môi trường, Arrow, evaluation và fine-tuning | Commit thật + checkpoint/log |
| Thành viên 3 | DSCMR | Feature đúng nguồn, train và kiểm tra protocol | Commit thật + feature manifest |
| Thành viên 4 | DCMH | MAT/VGG-F, port code cũ, các độ dài mã | Commit thật + config/log |
| Thành viên 5 | DCHMT | Converter, checkpoint CLIP, hashing experiments | Commit thật + config/log |

Các file trong gói ban đầu do trợ lý tạo. Nếu có bootstrap commit, tác giả là `Codex Assistant`, không phải 5 thành viên. Không tạo lịch sử commit giả hoặc chia commit tự động để chứng minh đóng góp chưa diễn ra.

Mỗi thành viên dùng Git identity của chính mình, commit phần mình đã thực hiện và giải thích được. Không sửa Git identity toàn cục của máy chỉ để chạy gói này. Repo upstream cần giữ nguồn/license và phân biệt với code nhóm.

Repository remote của nhóm chưa được cung cấp, vì vậy chưa push lên GitHub và chưa có link repo nộp bài. Sau khi tạo repo nhóm, có thể import lịch sử thật trong `bootstrap.bundle`; không coi bootstrap là phần thực nghiệm đã hoàn tất.
