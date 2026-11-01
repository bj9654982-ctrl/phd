from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from difflib import SequenceMatcher
import requests,json,re,sys,time,unicodedata,argparse
ROOT=Path(__file__).resolve().parents[1]
LIB=ROOT.parent/'v2_300篇'/'library_300.json'
CACHE=ROOT/'metadata_cache';CACHE.mkdir(parents=True,exist_ok=True)

def norm(s):return ''.join(c for c in unicodedata.normalize('NFKD',s).casefold() if c.isalnum())
def fetch(row, retry=False):
    p=CACHE/f"{row['number']:03d}.json"
    if p.exists():
        old=json.loads(p.read_text(encoding='utf-8'))
        if not retry or '429' not in old.get('error',''):return old
    out={'number':row['number'],'original_doi':row.get('doi',''),'matched':False}
    try:
        if row.get('doi'):
            r=requests.get('https://api.crossref.org/works/'+row['doi'],timeout=(10,35));r.raise_for_status();items=[r.json()['message']]
        else:
            year=int(row['year']);r=requests.get('https://api.crossref.org/works',params={'query.title':row['title'],'rows':3,'filter':f'from-pub-date:{year-1}-01-01,until-pub-date:{year+1}-12-31'},timeout=(10,35));r.raise_for_status();items=r.json()['message']['items']
        ranked=[]
        for it in items:
            title=(it.get('title') or [''])[0]
            score=SequenceMatcher(None,norm(title),norm(row['title'])).ratio()
            ranked.append((score,it))
        if ranked:
            score,it=max(ranked,key=lambda x:x[0]);out.update(score=score,candidate_title=(it.get('title') or [''])[0])
            if score>=.96:
                out.update(matched=True,metadata=it)
            else:out['note']='Title mismatch; do not automatically adopt'
    except Exception as e:out['error']=str(e)[:500]
    p.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');return out

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');lib=json.loads(LIB.read_text(encoding='utf-8'));out=[]
    if '--retry-errors' in sys.argv:
        for row in lib:
            old=json.loads((CACHE/f"{row['number']:03d}.json").read_text(encoding='utf-8'))
            if '429' in old.get('error',''):
                time.sleep(1.5)
                result=fetch(row,True)
                if '429' in result.get('error',''):time.sleep(20)
            else:result=old
            out.append(result)
            if len(out)%25==0:print('Slow retry',len(out),'matched',sum(x['matched'] for x in out),flush=True)
        (ROOT/'crossref_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
        print('Retry complete',sum(x['matched'] for x in out),flush=True)
        sys.exit(0)
    with ThreadPoolExecutor(max_workers=4) as ex:
        for fu in as_completed([ex.submit(fetch,r) for r in lib]):
            out.append(fu.result())
            if len(out)%25==0:print('Metadata checked',len(out),'matched',sum(x['matched'] for x in out),flush=True)
    (ROOT/'crossref_audit.json').write_text(json.dumps(sorted(out,key=lambda r:r['number']),ensure_ascii=False,indent=2),encoding='utf-8')
    print('Done',len(out),'matched',sum(x['matched'] for x in out),flush=True)
