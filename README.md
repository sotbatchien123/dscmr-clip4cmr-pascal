# SEG Assignment 2 - 5 phương pháp

**Trạng thái bàn giao: bộ mã hỗ trợ tái lập, CHƯA hoàn thành thực nghiệm Assignment 2.**
Không có điểm benchmark thực tế nào được sinh trong phiên này. Đọc `REPORT.md` trước khi sử dụng.

## Phạm vi

| Thư mục | Phương pháp | Dữ liệu | Nội dung đã soạn |
|---|---|---|---|
| `01_CLIP` | CLIP | Flickr30K | Evaluator hai chiều, lưu embedding và ví dụ top-10 |
| `02_ViLT` | ViLT | Flickr30K | Adapter train/eval chính thức và converter Arrow |
| `03_DSCMR` | DSCMR | Pascal Sentence | Driver dùng model/loss gốc, lưu checkpoint, hai chế độ chọn model |
| `04_DCMH` | DCMH | MIRFlickr-25K | Adapter bản PyTorch cộng đồng, cấu hình bit/epoch/batch |
| `05_DCHMT` | DCHMT | MIRFlickr-25K | Adapter repo gốc, kiểm tra các đầu vào cần thiết |

Code thuật toán gốc của các tác giả **chưa tải được và không nằm trong ZIP**. `fetch_upstream.py` chuẩn bị thao tác clone, giữ lại revision thật khi thành công. Các adapter chưa được chạy end-to-end, không được gọi là implementation đã xác nhận hoạt động.

## Bắt đầu trên Windows

Giải nén vào thư mục riêng, mở PowerShell tại thư mục có README này. Dùng Python 3.10 trở lên.

```powershell
$py = 'C:\Program Files\Python310\python.exe'
& $py scripts\preflight.py --out audit\environment_local.json
& $py -m unittest discover -s tests -v
& $py scripts\fetch_upstream.py --method all
```

Lệnh kiểm tra metric chỉ dùng dữ liệu tổng hợp; không phải huấn luyện hay kết quả dataset.
Nếu clone thất bại, đọc `results/upstream.json` trong từng phương pháp. Không khởi chạy model khi code hoặc dữ liệu chưa có.

## Môi trường riêng

Không cài đè các Python đang dùng cho dự án khác. Tạo venv riêng cho từng phương pháp nếu cần. Ví dụ môi trường CLIP/DSCMR để thử tương thích:

```powershell
& 'C:\Program Files\Python310\python.exe' -m venv .venv-clip
$py = (Resolve-Path .venv-clip\Scripts\python.exe).Path
& $py -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
& $py -m pip install -r requirements-common.txt
& $py -m pip install -e upstream\clip
```

Đây là cấu hình khởi điểm CHƯA kiểm chứng tại máy, không phải môi trường đã khóa cho cả 5 repo. Nguồn wheel: https://pytorch.org/get-started/previous-versions/ . Repo ViLT/DCMH cũ có thể cần port API trước khi chạy với PyTorch mới; không cài toàn bộ requirements cũ vào venv CLIP.

Chỉ train khi GPU đủ trống. Không chạy đồng thời 5 model trên RTX 5070 8 GB. Chụp cấu hình thực tế bằng preflight; không dùng thông số GPU để suy ra thời gian chạy chưa đo.

## Tiêu chuẩn một phương pháp hoàn thành

1. Có commit upstream và phần thay đổi của nhóm.
2. Dataset, split, nhãn/caption, preprocessing và hash manifest xác định được.
3. Có config, seed, dependencies, checkpoint và stdout thật.
4. Đánh giá hai chiều trên toàn bộ tập quy định; test subset được đánh dấu riêng.
5. Ghi đúng dòng/bảng paper gốc và so sánh chỉ khi protocol tương ứng.
6. Mỗi thành viên có commit công việc thực, không gán commit tự động thành đóng góp của người khác.

Mở README trong từng phương pháp để xem bước riêng. `CONTRIBUTIONS.md` mô tả phân công dự kiến, không xác nhận ai đã hoàn thành.
