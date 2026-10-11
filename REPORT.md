# Báo cáo trạng thái thực hiện SEG Assignment 2

Ngày kiểm tra: 11/10/2026 (Asia/Bangkok).

## Kết luận thực tế

**CHƯA hoàn thành Assignment 2. Không có kết quả benchmark của 5 mô hình.**

Đã chuẩn bị bộ mã hỗ trợ tái lập, tách 5 phương pháp; viết evaluator CLIP, driver DSCMR dựa trên model/loss tác giả, adapter gọi ViLT/DCMH/DCHMT, kiểm tra dữ liệu, công cụ lưu log/seed/revision và kiểm tra metric bằng dữ liệu tổng hợp.

## Điều kiện đã kiểm tra

- `nvidia-smi` lần đầu: NVIDIA GeForce RTX 5070 Laptop, 8151 MiB VRAM, dùng 6885 MiB, GPU utilization 98%. Đây là ảnh chụp tại thời điểm kiểm tra, không phải tình trạng cố định của máy.
- `C:\Program Files\Python310\python.exe`: Python 3.10.11, có NumPy/SciPy/Pillow, không có torch hoặc transformers. Python 3.9 được PATH chọn cũng không có torch.
- Kiểm tra lại lúc 17:34:44: GPU đã rảnh (1 MiB, utilization 0%). GPU bận không còn là điểm chặn tại lần kiểm tra này; thiếu runtime và đường tải code/dữ liệu vẫn còn.
- `git ls-remote https://github.com/penghu-cs/DSCMR.git HEAD` thất bại với `getaddrinfo() thread failed to start`.
- Không dừng tiến trình GPU đang chạy, không cài đè môi trường của dự án khác.
- Web đọc được README/source snippets nhưng không đồng nghĩa shell tải được repo/checkpoint/dataset.

Chi tiết đo và kiểm tra phần mềm được ghi trong `audit/`. Không dùng thông tin phần cứng để suy ra điểm hoặc thời gian training chưa đo.

## Đối chiếu yêu cầu giảng viên

| Yêu cầu Assignment 2 | Trạng thái | Còn thiếu |
|---|---|---|
| Implement một/nhiều thuật toán | Một phần: evaluator/driver/adapter, chưa tích hợp chạy end-to-end | Tải upstream, port dependency, thực thi model và xác minh đúng thuật toán |
| Repo Git | Bootstrap cục bộ nếu Git tạo thành công; xem audit Git | Remote repo và đóng góp thực tế của 5 người |
| Commits with contributions | Chưa đáp ứng cho 5 thành viên | Từng người thực hiện và commit công việc thật |
| Chạy dataset của paper | Chưa thực hiện | Dataset, preprocessing, split, checkpoint và môi trường |
| Reproduce kết quả paper | Chưa thực hiện | Kết quả chạy thật, dòng tham chiếu đã xác minh và phân tích sai lệch |

## Kết quả theo phương pháp

| Phương pháp | Dataset mục tiêu | Kết quả thực | Lý do chưa có |
|---|---|---|---|
| CLIP | Flickr30K | Chưa có R@K | Thiếu runtime, ảnh/caption/split và trọng số |
| ViLT | Flickr30K | Chưa có R@K | Thiếu repo/dependency tương thích, Arrow và checkpoint |
| DSCMR | Pascal Sentence | Chưa có mAP | Thiếu repo, feature tương ứng và torch |
| DCMH | MIRFlickr-25K | Chưa có mAP | Thiếu MAT/VGG-F, repo và port code cũ |
| DCHMT | MIRFlickr-25K | Chưa có mAP | Thiếu converter output, CLIP weights, repo/runtime |

Các giá trị trống trong bảng kết quả không phải 0. Không có số liệu sao chép từ paper trong cột kết quả của nhóm.

## Phát hiện khi rà soát code nguồn

1. DSCMR có vấn đề protocol cần khai báo: train_model chọn checkpoint dựa trên mAP của split tên `test`. Driver có mode `upstream-test` để giữ hành vi này và `fixed-last` để đánh giá sau epoch cuối; hai mode không phải cùng protocol. Cấu hình main và chữ ký train_model có thể khác nhau theo revision; dùng keyword epochs giúp tránh nhầm vị trí tham số.
2. DCMH PyTorch được đề xuất là repo cộng đồng. Bản tác giả gốc ưu tiên MATLAB. Chưa chứng minh kết quả hai bản tương đương.
3. ViLT trả IR trước TR trong ví dụ EVAL; phải ánh xạ IR=T2I và TR=I2T.
4. DCHMT repo cũ đã chuyển sang repo chung. Adapter dùng repo cũ và không trộn cấu hình YAML mới.
5. R@K truy hồi caption và mAP theo nhãn là hai protocol khác nhau. Không lập bảng xếp hạng chung từ các dataset/metric không tương đồng.

## Phần kiểm tra có thể chạy tại chỗ

Kiểm tra bằng dữ liệu tổng hợp: nhiều caption cho một ảnh, AP tính tay, query không có positive, tie Hamming, nhãn đa lớp, hash dài tránh tràn số và NaN. Các kiểm tra này là SOFTWARE TEST, không là thí nghiệm model. Xem `audit/software_checks.json` và `audit/tests.log` để biết số test và exit code thật.

## Phần tiếp tục khi có đủ điều kiện

1. Giải quyết truy cập Internet của môi trường thực thi; clone code và ghi commit SHA.
2. Chuẩn bị đúng 3 dataset và các feature/checkpoint phụ thuộc; không thay dữ liệu để lấy kết quả nhanh.
3. Tạo môi trường riêng cho từng phương pháp, xử lý code cũ theo lỗi thật.
4. Chạy smoke test trước, rồi chạy đầy đủ từng phương pháp khi GPU rảnh. Smoke test không đưa vào bảng benchmark.
5. Giữ output/log/checkpoint của từng run, đối chiếu đúng bảng trong bài gốc và ghi mọi thay đổi protocol.
6. Hoàn thiện repo nhóm và contributions. Chỉ lúc đó mới có thể nộp với tuyên bố hoàn thành Assignment 2.

## Nguồn

- Yêu cầu giảng viên: ảnh `image.png` do người dùng cung cấp, mục Assignment 2.
- Nền tảng chọn phương pháp: `SEG (1).pdf` và survey Bridging Modalities do người dùng cung cấp.
- https://github.com/openai/CLIP
- https://github.com/dandelin/ViLT/blob/master/DATA.md
- https://github.com/dandelin/ViLT/blob/master/EVAL.md
- https://github.com/dandelin/ViLT/blob/master/TRAIN.md
- https://github.com/penghu-cs/DSCMR/blob/master/main.py
- https://github.com/penghu-cs/DSCMR/blob/master/train_model.py
- https://github.com/penghu-cs/DSCMR/blob/master/load_data.py
- https://github.com/jiangqy/DCMH-CVPR2017
- https://github.com/WendellGul/DCMH
- https://github.com/kalenforn/DCHMT
- https://bryanplummer.com/Flickr30kEntities/
- https://vision.cs.uiuc.edu/pascal-sentences/
- https://press.liacs.nl/mirflickr/mirdownload.html

Nguồn web được kiểm tra ngày 11/10/2026. Các nhánh upstream chưa được pin vì chưa clone thành công; không có SHA được tự suy đoán.
