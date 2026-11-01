from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
import requests,json,re,sys
from selection_plan import CLASSIC_QUERIES
sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parent
def query(item):
    title,year=item
    r=requests.get('https://api.crossref.org/works',params={'query.title':title,'rows':3,'select':'title,author,published,container-title,volume,issue,page,DOI,type,URL'},timeout=(8,35))
    r.raise_for_status()
    return {'query':title,'expected_year':year,'results':r.json()['message']['items']}
out=[]
with ThreadPoolExecutor(max_workers=3) as pool:
    jobs={pool.submit(query,v):v for v in CLASSIC_QUERIES}
    for f in as_completed(jobs):
        try:out.append(f.result())
        except Exception as e:out.append({'query':jobs[f][0],'error':str(e)})
(ROOT/'classic_crossref_candidates.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
for r in out:
    print(r['query'])
    for v in r.get('results',[]):print(' ',v.get('title'),v.get('published'),v.get('DOI'))
    if 'error' in r:print(r['error'])
