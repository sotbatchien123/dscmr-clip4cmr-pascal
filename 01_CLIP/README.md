# 01 - CLIP

## Phần đã viết và giới hạn

`evaluate.py` dùng thư viện chính thức OpenAI CLIP, tính embedding chuẩn hóa và full-gallery similarity. Có R@1/5/10 hai chiều, embedding, gallery và 20 ví dụ T2I top-10. Chưa kiểm thử model vì thiếu PyTorch, checkpoint và Flickr30K.

- Code: https://github.com/openai/CLIP
- Paper: Learning Transferable Visual Models From Natural Language Supervision (2021), được survey dẫn ở [12].
- Dữ liệu: https://bryanplummer.com/Flickr30kEntities/
- Đối chiếu split: https://github.com/dandelin/ViLT/blob/master/DATA.md

## Dữ liệu

Tải ảnh theo điều kiện của nguồn Flickr30K, lấy `dataset_flickr30k.json` theo Karpathy split. Không tự chia lại khi muốn đối chiếu benchmark gốc. Test thường có 1.000 ảnh và 5.000 caption; kiểm tra cả danh sách ID, không chỉ số lượng.

```powershell
& $py scripts\validate_data.py flickr30k --annotations datasets\flickr30k\karpathy\dataset_flickr30k.json --images datasets\flickr30k\flickr30k-images
```

## Chạy

Sau khi clone và cài `upstream/clip` trong môi trường riêng:

```powershell
& $py 01_CLIP\evaluate.py --annotations datasets\flickr30k\karpathy\dataset_flickr30k.json --images datasets\flickr30k\flickr30k-images --model ViT-B/32 --device cuda --batch-size 16 --out 01_CLIP\results\full_test_seed42
```

`ViT-B/32` là điểm bắt đầu, không mặc định tương ứng với mọi số trong paper CLIP. Phải ghi rõ model của dòng tham chiếu. `--limit-images 20` chỉ để smoke test; gallery nhỏ làm bài toán dễ hơn và không được đưa vào bảng benchmark. Không có cắt ngắn caption mặc định; bật `--truncate` là một lựa chọn preprocessing phải ghi nhận.

## Kết quả cần nộp

- Hai chiều R@1, R@5, R@10 (JSON lưu dạng 0..1; nhân 100 khi trình bày phần trăm).
- Backbone/checkpoint, split và SHA256 annotations.
- Ví dụ đúng/sai, thời gian đo thật và môi trường.
- Ghi phạm vi là tái lập đánh giá zero-shot; không nói đã pretrain CLIP lại. Nếu giảng viên yêu cầu train cho từng người thì CLIP evaluation hiện chưa đáp ứng phần đó.
