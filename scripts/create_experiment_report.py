from pathlib import Path
import json,hashlib
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT  # repo root (was hard-coded D:/SEG)
OUT=ROOT/'reports';OUT.mkdir(exist_ok=True)
names=['DSCMR','CLIP4CMR'];new={};old={};hist={};protocol={};evidence={}
oldruns={'DSCMR':'pascal_validation_refit_seed42','CLIP4CMR':'pascal_uiuc_seed42'}
for name in names:
    folder=SOURCE/name/'runs/pascal_selection5000_refit1000_seed42'
    new[name]=json.loads((folder/'result.json').read_text())
    old[name]=json.loads((SOURCE/name/'runs'/oldruns[name]/'result.json').read_text())
    hist[name]=json.loads((folder/'history.json').read_text())
    protocol[name]=json.loads((folder/'protocol.json').read_text())
    refit=json.loads((folder/'refit_history.json').read_text())
    assert len(hist[name])==5000 and len(refit)==1000
    assert [r['epoch'] for r in hist[name]]==list(range(1,5001))
    assert [r['epoch'] for r in refit]==list(range(1,1001))
    assert json.loads((folder/'verification.json').read_text())['status']=='passed'
    evidence[name]={'new':new[name],'previous':old[name],
        'history_count':len(hist[name]),'refit_count':len(refit),
        'source_result':str(folder/'result.json'),
        'result_sha256':hashlib.sha256((folder/'result.json').read_bytes()).hexdigest()}
(OUT/'report_evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')

doc=Document();sec=doc.sections[0]
sec.page_width=Inches(8.5);sec.page_height=Inches(11)
sec.top_margin=sec.bottom_margin=Inches(.7)
sec.left_margin=sec.right_margin=Inches(.8)
normal=doc.styles['Normal'];normal.font.name='Calibri';normal.font.size=Pt(11)
normal.paragraph_format.space_after=Pt(7);normal.paragraph_format.line_spacing=1.12
for style,size in [('Title',22),('Heading 1',15),('Heading 2',12)]:
    doc.styles[style].font.name='Calibri';doc.styles[style].font.size=Pt(size)
    doc.styles[style].font.color.rgb=RGBColor(0,0,0)
    doc.styles[style].paragraph_format.space_before=Pt(10)
    doc.styles[style].paragraph_format.space_after=Pt(6)
footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.RIGHT
run=footer.add_run();field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');run._r.append(field)
doc.core_properties.title='Báo cáo thực nghiệm DSCMR và CLIP4CMR trên Pascal Sentence'
doc.core_properties.subject='Truy hồi ảnh và văn bản hai chiều với cấu hình 5000 và 1000 epoch'
doc.core_properties.author=''
for element in list(doc.styles.element.iter(qn('w:pBdr'))):
    element.getparent().remove(element)

def p(text): return doc.add_paragraph(text)
def h(text): doc.add_heading(text,level=1)
def sub(text): doc.add_heading(text,level=2)
def page(): doc.add_page_break()
def table(headers,rows,widths=None):
    t=doc.add_table(rows=1,cols=len(headers));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
    if widths is None: widths=[6.9/len(headers)]*len(headers)
    for c,width in zip(t.columns,widths): c.width=Inches(width)
    for cell,label in zip(t.rows[0].cells,headers):cell.text=label
    for row in rows:
        cells=t.add_row().cells
        for c,text in zip(cells,row): c.text=str(text)
    for ri,row in enumerate(t.rows):
        props=row._tr.get_or_add_trPr();keep=OxmlElement('w:cantSplit');props.append(keep)
        if ri==0: props.append(OxmlElement('w:tblHeader'))
        for ci,cell in enumerate(row.cells):
            cell.width=Inches(widths[ci]);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            tcpr=cell._tc.get_or_add_tcPr()
            shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'17365D' if ri==0 else ('EFF3F7' if ri%2 else 'FFFFFF'));tcpr.append(shade)
            borders=OxmlElement('w:tcBorders')
            for edge in ['top','left','bottom','right']:
                el=OxmlElement('w:'+edge);el.set(qn('w:val'),'single');el.set(qn('w:sz'),'4');el.set(qn('w:color'),'D9D9D9');borders.append(el)
            tcpr.append(borders)
            margin=OxmlElement('w:tcMar')
            for edge in ['top','left','bottom','right']:
                el=OxmlElement('w:'+edge);el.set(qn('w:w'),'90');el.set(qn('w:type'),'dxa');margin.append(el)
            tcpr.append(margin)
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_after=Pt(2);paragraph.paragraph_format.line_spacing=1.05
                paragraph.alignment=WD_ALIGN_PARAGRAPH.LEFT if ci==0 else WD_ALIGN_PARAGRAPH.CENTER
                for r in paragraph.runs:
                    r.font.size=Pt(10);r.bold=ri==0
                    if ri==0:r.font.color.rgb=RGBColor(255,255,255)
    doc.add_paragraph().paragraph_format.space_after=Pt(0)
def pct(x):return f'{100*x:.2f}'.replace('.',',')

doc.add_paragraph('Báo cáo thực nghiệm DSCMR và CLIP4CMR',style='Title')
p('Truy hồi ảnh và văn bản trên Pascal-Sentence\nNgày lập báo cáo 16/10/2026')
h('1 Mục tiêu và kết quả chính')
p('Thực nghiệm đánh giá hai chiều truy hồi: dùng ảnh để tìm văn bản (I2T) và dùng văn bản để tìm ảnh (T2I). Chỉ số chính là mAP@ALL; R@1, R@5 và R@10 được báo cáo bổ sung. Mục tiêu của lượt chạy mở rộng là quan sát ảnh hưởng của việc tăng giai đoạn validation lên 5.000 epoch và huấn luyện lại trên toàn bộ train trong 1.000 epoch.')
p('Cả hai mô hình đã hoàn tất cấu hình này. DSCMR đạt mAP 63,36% ở chiều I2T và 65,86% ở chiều T2I; CLIP4CMR đạt lần lượt 65,03% và 65,90%. Các giá trị đều thấp hơn lượt chạy trước trong cùng mô hình. Kết quả chưa cho phép xếp hạng trực tiếp hai mô hình vì DSCMR dùng 100 mẫu test, còn CLIP4CMR dùng 200 mẫu test với cách chuẩn bị dữ liệu khác.')
h('2 Mô hình và dữ liệu sử dụng')
sub('DSCMR')
p('DSCMR học biểu diễn chung từ hai nhánh ảnh và văn bản có giám sát bằng nhãn [1]. Trong mã thực nghiệm, đặc trưng ảnh 4.096 chiều và văn bản 300 chiều được đưa qua các nhánh riêng, sau đó qua lớp chiếu dùng chung để tạo biểu diễn 1.024 chiều. Hàm mất mát kết hợp dự đoán nhãn, quan hệ ngữ nghĩa và sự gần nhau của cặp ảnh–văn bản. Truy hồi sử dụng khoảng cách cosine [4].')
sub('CLIP4CMR')
p('CLIP4CMR sử dụng đặc trưng ảnh và văn bản 1.024 chiều từ CLIP RN50 cố định. Hai mạng chiếu riêng có kích thước 1024 → 256 → 1024, dùng GELU, dropout và chuẩn hóa L2. Nhánh loss PCL huấn luyện các biểu diễn theo prototype của lớp; bộ đánh giá sử dụng khoảng cách Euclidean [2, 5]. Thực nghiệm chỉ huấn luyện mạng truy hồi phía sau, không cập nhật trọng số CLIP.')
table(['Thuộc tính','DSCMR','CLIP4CMR'],[
 ['Nguồn đặc trưng','Có sẵn trong repo tác giả','Tự trích từ ảnh và caption UIUC'],
 ['Train / test','800 / 100','800 / 200'],
 ['Số lớp','20','20'],
 ['Ảnh / văn bản đầu vào','4096 / 300 chiều','1024 / 1024 chiều']], [2.0,2.2,2.7])

page();h('3 Chuẩn bị dữ liệu và quy trình chạy')
p('Dữ liệu DSCMR giữ nguyên các tệp đặc trưng trong repo. Bộ CLIP4CMR được xây dựng thay thế vì link đặc trưng tác giả không sử dụng được tại thời điểm chuẩn bị. Nguồn UIUC cung cấp 1.000 ảnh thuộc 20 lớp [6]. Mỗi lớp được chia 40 ảnh train và 10 ảnh test bằng seed 42, tổng cộng 800/200. ID và mã SHA256 được lưu; kiểm tra không phát hiện trùng ID hoặc trùng nội dung tệp ảnh giữa hai tập.')
p('Với CLIP4CMR, toàn bộ caption có sẵn của một ảnh được ghép theo thứ tự nguồn. Bộ tokenize của CLIP cắt đầu vào theo giới hạn 77 token; không tự tạo thêm caption bị thiếu. Ảnh và toàn bộ caption của nó luôn thuộc cùng một split. Đây là quy tắc local, chưa xác minh trùng cách xử lý của tác giả.')
sub('Hai giai đoạn huấn luyện')
p('Giai đoạn 1 tách phân tầng 80 mẫu validation từ 800 mẫu train, còn 720 mẫu dùng cập nhật trọng số. Chạy đủ 5.000 epoch, đánh giá validation sau mỗi epoch và lưu checkpoint có mAP trung bình hai chiều cao nhất. Không dừng sớm. Checkpoint tốt nhất nằm tại epoch 61 với DSCMR và epoch 69 với CLIP4CMR.')
p('Giai đoạn 2 khởi tạo lại mô hình với cùng seed, rồi huấn luyện đúng 1.000 epoch trên toàn bộ 800 mẫu train, bao gồm 80 mẫu từng dùng validation. Cuối cùng đánh giá checkpoint epoch 1.000 trên test. Số epoch tốt nhất của giai đoạn 1 chỉ được ghi nhận, không quyết định thời lượng hoặc khởi tạo của giai đoạn 2. Tổng 6.000 epoch là tổng hai lượt khởi tạo riêng.')
table(['Thông số','DSCMR','CLIP4CMR'],[
 ['Optimizer','Adam','Adam'],['Learning rate','0,0001','0,0001'],['Adam betas','0,5 và 0,999','0,5 và 0,999'],['Batch size','100','300'],['Seed','42','42'],['Validation / refit','5000 / 1000 epoch','5000 / 1000 epoch']], [2.5,2.2,2.2])
h('4 Chỉ số đánh giá')
p('Một kết quả được xem là phù hợp khi có cùng nhãn lớp với truy vấn. I2T xếp hạng toàn bộ văn bản test cho mỗi ảnh test; T2I xếp hạng toàn bộ ảnh test cho mỗi văn bản test. Mỗi chiều dùng 100 truy vấn và 100 ứng viên với DSCMR, hoặc 200 truy vấn và 200 ứng viên với CLIP4CMR.')
p('AP của một truy vấn được tính bằng tổng precision tại các vị trí có kết quả đúng, chia cho tổng số kết quả phù hợp trong tập ứng viên. mAP@ALL là trung bình AP trên tất cả truy vấn, xét toàn bộ danh sách xếp hạng. R@K trong báo cáo là tỷ lệ truy vấn có ít nhất một kết quả phù hợp trong K vị trí đầu; đây là hit rate theo truy vấn, không phải tỷ lệ thu hồi toàn bộ tài liệu liên quan.')

page();h('5 Kết quả thực nghiệm')
p('Bảng dưới đây trình bày kết quả test của mô hình sau 1.000 epoch refit. Tất cả chỉ số đều tính theo phần trăm. Hai mô hình được đặt cùng bảng để trình bày kết quả, không phải so sánh trên cùng một tập test.')
rows=[]
for name in names:
 for direction in ['I2T','T2I']:
    m=new[name]['metrics'][direction];rows.append([name,direction,*[pct(m[k]) for k in ['mAP@ALL','R@1','R@5','R@10']]])
table(['Mô hình','Chiều','mAP@ALL','R@1','R@5','R@10'],rows,[1.45,.65,1.2,1.2,1.2,1.2])
sub('Ảnh hưởng của việc tăng số epoch')
p('Lượt trước chọn thời lượng refit theo validation: DSCMR chạy 500 epoch theo dõi validation rồi refit 61 epoch; CLIP4CMR chạy 200 epoch rồi refit 69 epoch. Lượt mới giữ nguyên dữ liệu, seed, loss, learning rate và batch nhưng dùng lịch 5000/1000. Vì dữ liệu của từng mô hình không đổi, có thể đối chiếu trước–sau trong chính mô hình đó.')
rows=[]
for name in names:
 for direction in ['I2T','T2I']:
    a=old[name]['metrics'][direction]['mAP@ALL'];b=new[name]['metrics'][direction]['mAP@ALL']
    rows.append([name,direction,pct(a),pct(b),f'{100*(b-a):+.2f}'.replace('.',',')])
table(['Mô hình','Chiều','mAP trước','mAP mới','Chênh lệch đpt'],rows,[1.5,.65,1.5,1.5,1.75])
declines={name:sum((old[name]['metrics'][d]['mAP@ALL']-new[name]['metrics'][d]['mAP@ALL'])*100 for d in ['I2T','T2I'])/2 for name in names}
p(('Đpt là điểm phần trăm. Mức giảm trung bình hai chiều là %.2f điểm với DSCMR và %.2f điểm với CLIP4CMR. Việc kéo dài huấn luyện đến 1.000 epoch chưa mang lại cải thiện trên hai tập test đang dùng.' % (declines['DSCMR'],declines['CLIP4CMR'])).replace('2.60','2,60').replace('6.04','6,04'))
p('Trong toàn bộ 5.000 epoch, checkpoint validation tốt nhất vẫn ở epoch 61 và 69. Diễn biến này phù hợp với khả năng mô hình học quá lâu và giảm khả năng tổng quát hóa. Tuy nhiên, một seed và một split chưa đủ để khẳng định nguyên nhân; cần nhiều lần chạy và phân tích đường loss trước khi kết luận chắc chắn về overfitting.')
sub('Kiểm chứng kết quả')
p('Đã kiểm tra các tệp history.json có đủ 5.000 epoch liên tiếp và refit_history.json có đủ 1.000 epoch. Các chỉ số lưu trong result.json đã được tính lại từ test_features.npz; mAP hai chiều khớp bộ đánh giá của tác giả. Không dùng test để chọn checkpoint trong runner. Checkpoint cũ và mới đều được giữ riêng để truy nguyên.')

page();h('6 Đối chiếu nguồn và giới hạn')
p('Survey Bridging Modalities tổng hợp mốc mAP@ALL dưới đây [3]. Bài CLIP4CMR xác nhận mốc Pascal-Sentence dùng 800 train/200 test và kết quả PCL tương ứng [2]. Các giá trị này là kết quả công bố, không phải kết quả đo trên máy local.')
table(['Mô hình','I2T công bố','T2I công bố'],[['DSCMR','67,4%','68,2%'],['CLIP4CMR','69,8%','69,2%']],[2.5,2.2,2.2])
p('Không diễn giải chênh lệch với bảng công bố như sai số tái lập: DSCMR local chỉ có 100 mẫu test; CLIP4CMR dùng split và quy tắc caption tự dựng. Nguồn gốc quá trình tạo đặc trưng DSCMR chưa được kiểm toán đầy đủ. Các yếu tố này ảnh hưởng trực tiếp đến độ khó của truy hồi.')
p('Thực nghiệm chỉ dùng một seed. Tập test đã được xem ở lượt trước nên lượt chạy mở rộng không phải kiểm định trên dữ liệu hoàn toàn chưa quan sát. Dù runner không chọn checkpoint bằng test, việc thử tiếp trên cùng test vẫn giới hạn mức độ độc lập của đánh giá. Thời gian chạy được ghi trong JSON nhưng không dùng để so tốc độ hai mô hình do chúng từng chạy đồng thời và có đầu vào khác nhau.')
h('7 Kết luận')
p('Cấu hình 5.000 epoch validation và 1.000 epoch refit đã hoàn thành cho cả hai mô hình. Trên dữ liệu hiện có, mAP test giảm so với lịch refit 61 và 69 epoch. Vì vậy chưa có căn cứ chọn lịch 1.000 epoch chỉ vì huấn luyện lâu hơn. Hướng tiếp theo là dùng validation để quyết định thời lượng huấn luyện, chạy thêm seed và chuẩn hóa cùng split test trước khi đánh giá mô hình nào hiệu quả hơn.')
h('Tài liệu tham khảo và bằng chứng')
refs=[
 '[1] Zhen và cộng sự. Deep Supervised Cross-Modal Retrieval. CVPR 2019. Mã chính thức: https://github.com/penghu-cs/DSCMR',
 '[2] Zeng và Mao. A Comprehensive Empirical Study of Vision-Language Pre-trained Model for Supervised Cross-Modal Retrieval. 2022. https://arxiv.org/html/2201.02772v2',
 '[3] Bridging Modalities: A Survey of Cross-Modal Image-Text Retrieval. Bảng 2. https://www.icck.org/article/html/cjif.2024.361895',
 '[4] DSCMR, model.py và train_model.py trong repo chính thức; bản mã local được dùng bởi scripts/train_selected.py.',
 '[5] CLIP4CMR, model.py, metrics.py và extract_clip_feature.py. https://github.com/zhixiongz/CLIP4CMR',
 '[6] UIUC Pascal Sentence Dataset. https://vision.cs.uiuc.edu/pascal-sentences/',
 '[7] Bằng chứng local: mỗi thư mục DSCMR/runs/ và CLIP4CMR/runs/ có lượt pascal_selection5000_refit1000_seed42, gồm result.json, protocol.json, history.json, refit_history.json và verification.json. Bản tổng hợp kiểm chứng: reports/verification_5000_1000.json.']
for text in refs:
    par=p(text);par.paragraph_format.space_after=Pt(4)
    for r in par.runs:r.font.size=Pt(9)
doc.save(OUT/'Bao_cao_DSCMR_CLIP4CMR.docx')
print('Created reports/Bao_cao_DSCMR_CLIP4CMR.docx')
