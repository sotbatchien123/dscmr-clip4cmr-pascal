# 04 - DCMH

- Paper: Deep Cross-Modal Hashing, CVPR 2017.
- Code tác giả: https://github.com/jiangqy/DCMH-CVPR2017 (tác giả ưu tiên MATLAB).
- Adapter ở đây chạy bản **cộng đồng**: https://github.com/WendellGul/DCMH .
- Dữ liệu gốc: https://press.liacs.nl/mirflickr/mirdownload.html .

Repo cộng đồng rất cũ (README ghi PyTorch 0.3+), CHƯA được port/kiểm chứng chạy trên PyTorch mới. Không có thuật toán DCMH upstream trong ZIP vì clone chưa thành công.

## Dữ liệu và trọng số

Repo cộng đồng cần `FLICKR-25K.mat` đã xử lý và `imagenet-vgg-f.mat`. README có link Baidu và mã truy cập; chưa xác minh tải được trong phiên này. Bộ ảnh gốc tải trực tiếp không tự thay thế hai file này.

Có thể dùng `scripts/download_mirflickr.py` lấy ảnh/tag/annotations gốc, rồi tái tạo đúng xử lý của phương pháp nếu không lấy được MAT. Bước tái tạo đó CHƯA được viết/kiểm chứng trong gói này; không dùng cách chia hoặc tag-vocabulary tùy ý để đối chiếu điểm công bố.

## Chạy sau khi giải quyết đầu vào và tương thích

```powershell
& $py 04_DCMH\run.py --data datasets\dcmh\FLICKR-25K.mat --vgg-weights weights\imagenet-vgg-f.mat --bits 16 --dry-run
```

Bỏ dry-run để chạy thật. Mặc định wrapper giữ batch 128, 500 epoch của bản cộng đồng. Chạy 16 bit trước, rồi 32/64 bit với cùng protocol. Kết quả chạy ở batch/epoch khác phải được ghi thành thiết lập khác.

## Đánh giá

Hamming ranking, hai chiều, mAP theo metric của paper/implementation được chọn. Kiểm tra AP@ALL hay AP@K, zero-positive queries, cách xử lý tie. `common/metrics.py` chỉ là bộ kiểm tra độc lập với tie stable; không tự coi nó bằng mọi evaluator gốc.

Code adapter lưu log; model outputs theo repo nằm trong `upstream/dcmh/checkpoints` và `result`. Cần lưu riêng mỗi run để tránh upstream ghi đè. Không được dùng số trong survey làm kết quả nhóm.
