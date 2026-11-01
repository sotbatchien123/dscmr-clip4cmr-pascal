import argparse,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def show():
    for name in ('DSCMR','CLIP4CMR'):
        p=ROOT/name/'status.json'
        d=json.loads(p.read_text(encoding='utf-8')) if p.exists() else {'state':'not_started'}
        print(name, ':', d.get('state',d.get('status')),flush=True)
        for direction,m in d.get('metrics',{}).items(): print(' ',direction,'mAP@ALL:',round(m['mAP@ALL']*100,4),'%',flush=True)
        if 'error' in d: print(' ',d['error'],flush=True)
        if 'epoch' in d: print(' ',d['epoch'],'/',d.get('total_epochs','?'),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',type=float,default=0);a=parser.parse_args()
    while True:
        show()
        if not a.watch: break
        time.sleep(a.watch)
