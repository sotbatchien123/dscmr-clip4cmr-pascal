# 02 - ViLT

Adapter dùng repo https://github.com/dandelin/ViLT . Không chứa lại thuật toán gốc. Chưa chạy được dependency/model trong phiên này.

## Chuẩn bị

- Paper: ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision (ICML 2021).
- Dataset: Flickr30K, dùng nguồn và split như CLIP.
- Repo `DATA.md`, `TRAIN.md`, `EVAL.md` là nguồn lệnh. README repo có checkpoint pretrained và checkpoint retrieval Flickr30K.
- Tạo môi trường riêng; kiểm tra API PyTorch Lightning, transformers, timm, pyarrow theo revision thực tế. Không giả định requirements năm 2021 chạy được nguyên trạng trên GPU mới.

```powershell
& $py 02_ViLT\prepare_data.py --upstream upstream\vilt --root datasets\flickr30k --output datasets\flickr30k_arrow
& $py 02_ViLT\run.py --arrow-root datasets\flickr30k_arrow --checkpoint weights\vilt_irtr_f30k.ckpt --mode evaluate --dry-run
```

Bỏ `--dry-run` để thực thi sau khi có môi trường/data/checkpoint. Mode `evaluate` cần checkpoint retrieval; mode `finetune` bắt đầu từ checkpoint pretraining:

```powershell
& $py 02_ViLT\run.py --arrow-root datasets\flickr30k_arrow --checkpoint weights\vilt_200k_mlm_itm.ckpt --mode finetune --batch-size 1
```

Batch 1 là điểm khởi đầu vì tài nguyên, không phải cam kết giữ nguyên effective batch của bài. Xác minh gradient accumulation, tổng bước, scheduler trước khi gọi là tái lập training. Sau train phải đánh giá checkpoint do mình train, không nhầm với checkpoint tác giả tải về.

## Báo cáo

Repo EVAL ghi thứ tự tuple: IR R@1/5/10 trước, TR R@1/5/10 sau. IR là T2I, TR là I2T. Giữ đúng thứ tự để tránh đảo hai chiều. Các số ví dụ trong README tác giả không phải số của nhóm.

`run.py` ghi command và stdout vào thư mục run riêng. Kết quả phải lấy từ run thật, lưu checkpoint, freeze môi trường và xác minh split. Runtime compatibility và Windows execution còn chưa được kiểm chứng.
