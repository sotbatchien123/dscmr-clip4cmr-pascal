> Cập nhật: CLIP4CMR đã chạy xong bằng nguồn UIUC thay thế. Các mô tả thiếu dữ liệu bên dưới là tình trạng trước phương án này. Xem báo cáo SELECTED_MODELS_REPORT.md.

# CLIP4CMR: kết quả tìm kiếm và kiểm tra ngày 14/10/2026

Đã tìm được nguồn Pascal-Sentence gốc còn truy cập được để chuẩn bị phương án tự trích xuất đặc trưng. Chưa lấy được hai tệp đặc trưng/split chính xác của CLIP4CMR; số epoch CLIP4CMR vẫn là 0. Chưa thể tuyên bố tái lập đúng số liệu survey.

## Những gì đã xác minh

| Hạng mục | Kết quả |
|---|---|
| Repo tác giả | https://github.com/zhixiongz/CLIP4CMR |
| Nhánh chính kiểm tra | main, commit d8b39012e3a85cc4828f06a7b00167f6cbcad3c5 |
| GitHub Releases | API trả danh sách rỗng; chưa tìm thấy checkpoint Pascal được công bố tại Releases |
| Link dữ liệu trong README | https://pan.cstcloud.cn/s/JqKbqGfTRs |
| Trạng thái link | Kiểm tra lại: ERR_SHARE_EXPIRED |
| Repo AdvCLIP dùng dữ liệu CLIP4CMR | README dẫn về đúng link hết hạn trên, không phải bản sao thay thế |
| Dữ liệu thô Pascal-Sentence | Trang UIUC còn truy cập được; đã đọc danh mục 1.000 ảnh, 20 lớp, mỗi lớp 50 ảnh |
| Tải ảnh | Đã tải và kiểm tra một JPEG 500 × 278; chưa tải toàn bộ 1.000 ảnh |
| Caption trên trang hiện tại | Đếm được 4.998 dòng: 996 ảnh có 5 caption, 3 ảnh có 4, 1 ảnh có 6 |
| Split và cách gom caption của CLIP4CMR | Chưa khôi phục được |

Nguồn dữ liệu: [UIUC Pascal Sentence](https://vision.cs.uiuc.edu/pascal-sentences/). Các ID bất thường về số caption: 2008_003858, 2008_008748, 2008_007742, 2008_006210. Không tự bù hoặc xoá caption. Trang được lưu và tính SHA256 trong evidence.json.

Đã lưu danh mục ID, lớp, URL ảnh và toàn bộ caption đã đọc vào `clip4cmr_investigation_20261014/pascal_raw_manifest.json`. Danh mục này chưa có nhãn train/test và chưa phải hai tệp đặc trưng tác giả.

## CLIP4CMR thực sự huấn luyện phần nào

Tác giả xác nhận trong [issue 5](https://github.com/zhixiongz/CLIP4CMR/issues/5#issuecomment-1097564050) rằng trọng số CLIP được cố định, chỉ dùng đặc trưng CLIP cho mạng truy hồi. Vì vậy không cần huấn luyện lại CLIP từ đầu để chạy thí nghiệm này.

Theo [mã trích xuất gốc](https://github.com/zhixiongz/CLIP4CMR/blob/main/extract_clip_feature.py), backbone là OpenAI CLIP RN50. Đặc trưng ảnh và văn bản đều có 1.024 chiều. `clip_train.pkl`/`clip_test.pkl` gồm image, text, label và ids. Các pkl là dữ liệu đặc trưng, không phải checkpoint mô hình.

Theo [model.py](https://github.com/zhixiongz/CLIP4CMR/blob/main/model.py), mỗi nhánh chiếu 1024 → 256 → 1024, dùng GELU, dropout 0,1 và chuẩn hoá L2 đầu ra. Loss PCL dùng hàm PAN cho cả hai nhánh với prototype theo lớp. Trong mã này, centers có requires_grad=False. Evaluator dùng khoảng cách Euclidean và cùng lớp là phù hợp.

Luồng chạy: ảnh/caption → CLIP RN50 cố định → hai vector 1024 chiều → mạng chiếu được huấn luyện bằng nhãn → xếp hạng I2T và T2I → mAP@ALL.

## Cấu hình bài báo và những chỗ mã nguồn khác

[Bài báo, mục 4.1](https://arxiv.org/html/2201.02772v2): Pascal-Sentence có 800 cặp train, 200 cặp test, 20 lớp; learning rate 1e-4, batch 300, dropout 0,1, tối đa 200 epoch, early stopping 20. Mốc PCL: I2T mAP 69,8%, T2I mAP 69,2%. Đây là số công bố, không phải số đo local.

Kiểm tra `main.py` trong repo cho thấy:

- Mặc định đang chọn nus-wide và 10 lớp; Pascal cần cấu hình lại 20 lớp.
- MAX_EPOCH=500 và early_stop=10, khác phần mô tả bài báo.
- Chỉ định cuda:3 và DataParallel trên GPU [3,1]; không chạy nguyên trạng trên một GPU.
- choose_loss mặc định là PAN nhưng nhánh thực thi có tên PCL; phải chọn đúng nhánh.
- `metrics.py` có docstring chưa đóng tại Proxy_NCA; adapter local đã nạp riêng nguyên hàm PAN cần dùng.
- Mã gốc đánh giá test sau mỗi epoch và dùng test chọn checkpoint. Runner local tách validation trong train; vì vậy cần ghi rõ khác protocol.

Không nên đồng nhất “đã có mã tác giả” với “chạy nguyên trạng là tái lập đúng bảng”.

## Vấn đề chưa giải quyết khi tự dựng dữ liệu

1. Trang UIUC có ảnh và nhiều caption, trong khi thí nghiệm có 1.000 cặp ảnh-văn bản. Chưa xác minh tác giả chọn một caption, ghép caption hay xử lý theo quy tắc khác.
2. Tệp `load_original_data` đọc danh sách split từ đường dẫn riêng trên máy tác giả. Ví dụ đang dùng wiki_image và văn bản JSON; nó chưa cung cấp bộ ID Pascal 800/200 cần thiết.
3. Script đó cắt 50 từ, bỏ dấu câu/chữ số rồi tokenize; trường hợp lỗi được bỏ qua. Chưa có bằng chứng toàn bộ quy trình này được áp dụng giống hệt cho Pascal.
4. DSCMR local hiện chỉ có 100 mẫu test với đặc trưng sẵn, chưa có ID ảnh. Không được lấy split mới của CLIP4CMR rồi gọi đó là cùng tập test với DSCMR.

## Các hướng tiếp tục có căn cứ

**Ưu tiên tái lập đúng mốc survey:** xin lại clip_train.pkl/clip_test.pkl Pascal, ID train/test, cách xử lý caption và cấu hình dùng ra bảng. Tác giả đã công khai hướng liên hệ `zengzhixiong2018@ia.ac.cn` tại [issue 4](https://github.com/zhixiongz/CLIP4CMR/issues/4#issuecomment-1084325181). Chưa gửi email hoặc đăng issue thay người dùng.

**Phương án thí nghiệm local:** từ nguồn UIUC đã tìm, có thể tải ảnh, quy định một split 800/200 có seed cố định và quy tắc văn bản, trích đặc trưng RN50 rồi huấn luyện mạng truy hồi. Đây là phương án khả thi dựa trên cấu trúc mã, chưa phải kết quả chạy. Phải đặt tên/split riêng và ghi khác dữ liệu tác giả; chưa dùng để kết luận hơn/kém survey hoặc DSCMR hiện tại.

Repo FedCMR còn chứa các tệp Pascal .mat, nhưng chưa xác minh chúng có CLIP RN50 features, đúng ID/split và cách tạo văn bản. Không nạp chúng thay hai pkl của CLIP4CMR chỉ vì trùng tên dataset.

## Bằng chứng tại máy

- `reports/clip4cmr_investigation_20261014/evidence.json`: kiểm tra link, số lượng mẫu, hash và giới hạn xác minh.
- `reports/clip4cmr_investigation_20261014/pascal_raw_manifest.json`: danh mục dữ liệu thô.
- `reports/clip4cmr_investigation_20261014/clip4cmr_issue_*_comments.json`: phản hồi công khai từ tác giả đã tải.
- `CLIP4CMR/upstream/`: mã gốc đã kiểm tra.
- Trạng thái lúc điều tra (14/10/2026): chưa có dữ liệu để chạy CLIP4CMR. Sau đó dữ liệu đã được dựng lại, xem mục 4 của README.
