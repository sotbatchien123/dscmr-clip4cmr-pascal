"""Create a local evidence record; do not create an unverified benchmark split."""
import collections,hashlib,html,json,re,shutil
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'reports/clip4cmr_investigation_20261014'
out.mkdir(parents=True,exist_ok=True)
source=ROOT/'tmp/pascal_sentences_source.html'
text=source.read_text(encoding='utf-8')
records=[]
for path,body in re.findall(r'<img src="([^"]+)"></td>\s*<td><table>(.*?)</table>',text,re.S):
    captions=[html.unescape(re.sub('<[^>]+>','',s)).strip() for s in re.findall(r'<tr><td>(.*?)</td></tr>',body,re.S)]
    records.append({'id':Path(path).stem,'class':path.split('/')[0],
                    'image_url':'https://vision.cs.uiuc.edu/pascal-sentences/'+path,'captions':captions})
assert len(records)==1000 and len({r['id'] for r in records})==1000
print('Caption counts:',collections.Counter(len(r['captions']) for r in records))
print('Unexpected rows:',[r for r in records if len(r['captions'])!=5][:2])
counts=dict(collections.Counter(r['class'] for r in records))
assert len(counts)==20 and set(counts.values())=={50}
with Image.open(ROOT/'tmp/pascal_sample.jpg') as image:
    sample={'format':image.format,'size':image.size};image.verify()
(out/'pascal_raw_manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
evidence={'date':'2026-10-14','official_repo_commit':'d8b39012e3a85cc4828f06a7b00167f6cbcad3c5',
    'cloud_link_response':json.loads((ROOT/'tmp/clip4cmr_link_check_new.json').read_text()),
    'official_releases':json.loads((ROOT/'tmp/clip4cmr_releases.json').read_text()),
    'raw_source':'https://vision.cs.uiuc.edu/pascal-sentences/',
    'raw_page_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
    'images_in_manifest':len(records),'captions':sum(len(r['captions']) for r in records),
    'classes':counts,'downloaded_sample':sample,'all_images_downloaded':False,
    'author_split_recovered':False,'author_caption_aggregation_recovered':False,
    'CLIP4CMR_training_epochs':0}
(out/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
for name in ['clip4cmr_issue_1_comments.json','clip4cmr_issue_4_comments.json','clip4cmr_issue_5_comments.json','clip4cmr_issue_6_comments.json','clip4cmr_issue_9_comments.json','clip4cmr_branches.json','clip4cmr_issues_new.json']:
    shutil.copy2(ROOT/'tmp'/name,out/name)
print(json.dumps(evidence,ensure_ascii=False,indent=2))
