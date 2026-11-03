# Ghi nhận bên thứ ba (Third-party notices)

File [LICENSE](LICENSE) (MIT) chỉ áp dụng cho mã nguồn và tài liệu do dự án viết. Các thành phần dưới đây thuộc bên thứ ba, giữ nguyên điều khoản của họ và **không được đưa vào repo**. Mọi thứ được tải lại khi chạy.

## Mã nguồn gốc của hai phương pháp

| Thành phần | Nguồn | License | Cách dùng trong dự án |
|---|---|---|---|
| DSCMR (Zhen et al., CVPR 2019) | [penghu-cs/DSCMR](https://github.com/penghu-cs/DSCMR) | Repo không khai báo license (mặc định giữ toàn quyền) | Tải về `DSCMR/upstream/` bằng `scripts/data/fetch_upstream.py`. Script của dự án đọc định nghĩa mạng và hàm loss từ thư mục này lúc chạy. Không phân phối lại. |
| CLIP4CMR (Zeng et al., 2022) | [zhixiongz/CLIP4CMR](https://github.com/zhixiongz/CLIP4CMR) | Repo không khai báo license (mặc định giữ toàn quyền) | Tải về `CLIP4CMR/upstream/`; xử lý như DSCMR. |

## Thư viện

| Thư viện | License |
|---|---|
| [PyTorch](https://github.com/pytorch/pytorch), torchvision | BSD-3-Clause |
| NumPy, SciPy | BSD-3-Clause |
| [OpenAI CLIP](https://github.com/openai/CLIP) | MIT |
| [open_clip](https://github.com/mlfoundations/open_clip) | MIT-style (xem repo) |
| [timm](https://github.com/huggingface/pytorch-image-models) | Apache-2.0 |
| Pillow, h5py, matplotlib, python-docx | Giấy phép riêng của từng thư viện (đều cho phép dùng tự do) |

## Trọng số mô hình tiền huấn luyện

| Mô hình | Nguồn | License |
|---|---|---|
| CLIP RN50, ViT-L/14, ViT-L/14-336 | OpenAI CLIP | MIT |
| EVA02-L/14-336 (merged2b_s6b_b61k) | [timm/eva02_large_patch14_clip_336.merged2b_s6b_b61k](https://huggingface.co/timm/eva02_large_patch14_clip_336.merged2b_s6b_b61k) | MIT |
| SigLIP-SO400M/14-384 (webli) | [timm/ViT-SO400M-14-SigLIP-384](https://huggingface.co/timm/ViT-SO400M-14-SigLIP-384) | Apache-2.0 |
| DFN5B CLIP ViT-H/14 | [apple/DFN5B-CLIP-ViT-H-14](https://huggingface.co/apple/DFN5B-CLIP-ViT-H-14) | Apple Sample Code License (apple-amlr). Kiểm tra điều khoản trước khi dùng ngoài mục đích nghiên cứu. |
| VGG19 (bản Caffe gốc của Oxford, qua Keras) | [fchollet/deep-learning-models](https://github.com/fchollet/deep-learning-models/releases/tag/v0.1) | Trọng số VGG gốc phát hành theo CC BY 4.0. Chỉ dùng cho thử nghiệm khôi phục mã ảnh (kết quả âm tính). |

Trọng số được tải vào cache cục bộ (Hugging Face cache, thư mục `checkpoints/`); không có trong repo.

## Dữ liệu

| Dữ liệu | Nguồn | Điều khoản |
|---|---|---|
| Pascal Sentence (Rashtchian et al., 2010) | [vision.cs.uiuc.edu/pascal-sentences](https://vision.cs.uiuc.edu/pascal-sentences/) | Câu mô tả: Creative Commons Attribution (trang gốc ghi Attribution-ShareAlike, liên kết tới CC BY 3.0). Ảnh lấy từ PASCAL VOC / Flickr, thuộc bản quyền của người chụp; chỉ dùng cho nghiên cứu. |
| Đặc trưng Pascal Sentence của DSCMR | thư mục `data/pascal` trong repo DSCMR | Theo repo DSCMR (không khai báo license) |
| Split và đặc trưng CLIP của tác giả CLIP4CMR (`clip_train.pkl`, `clip_test.pkl`) | Bản sao của bên thứ ba: HF Space [shivangibithel/LCMI_T2I](https://huggingface.co/spaces/shivangibithel/LCMI_T2I) | Space khai báo AFL-3.0; dữ liệu có nguồn gốc từ tác giả CLIP4CMR, chưa được tác giả xác nhận |

**Ngoại lệ duy nhất:** `reports/clip4cmr_investigation_20261014/pascal_raw_manifest.json` chứa mã ảnh, đường dẫn ảnh trên trang UIUC và 5.000 câu mô tả của Pascal Sentence. File này dùng để tải lại dữ liệu. Các câu mô tả trong file **không** thuộc MIT, mà thuộc giấy phép Creative Commons của tác giả bộ dữ liệu (Rashtchian et al., 2010; ghi công như phần Trích dẫn). File không chứa ảnh.

Ngoài file trên, dự án không phân phối lại dữ liệu nào ở trên. Cách tải lại và kiểm chứng được mô tả trong README và trong các script `scripts/verify_*.py`.

## Trích dẫn

- L. Zhen, P. Hu, X. Wang, D. Peng. *Deep Supervised Cross-modal Retrieval.* CVPR 2019.
- Z. Zeng, W. Mao. *A Comprehensive Empirical Study of Vision-Language Pre-trained Model for Supervised Cross-Modal Retrieval* (CLIP4CMR). arXiv:2201.02772, 2022.
- C. Rashtchian, P. Young, M. Hodosh, J. Hockenmaier. *Collecting Image Annotations Using Amazon's Mechanical Turk.* NAACL HLT 2010 Workshop.
