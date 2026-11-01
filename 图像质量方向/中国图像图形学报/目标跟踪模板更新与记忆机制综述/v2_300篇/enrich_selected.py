from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
import json, re, sys
from harvest_literature import fetch
from selection_plan import GROUPS

sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parent
rows=json.loads((ROOT/'candidates.json').read_text(encoding='utf-8'))
selected={i:k for k,values in GROUPS.items() for i in values}
out=[]
def enrich(row):
    row=dict(row)
    row['category']=selected[row['candidate_id']]
    try:
        soup=BeautifulSoup(fetch(row['url']),'lxml')
        element=soup.select_one('#abstract')
        if element is None:
            heading=soup.find(lambda t:t.name in ('h1','h2','h3') and t.get_text(strip=True).lower()=='abstract')
            if heading: element=heading.find_next('div') or heading.find_next('p')
        row['abstract']=' '.join(element.get_text(' ',strip=True).split()).strip('"') if element else ''
        row['landing_title']=' '.join(soup.select_one('#papertitle').get_text(' ',strip=True).split()).strip('"') if soup.select_one('#papertitle') else ''
        row['verification']='official metadata + abstract' if row['abstract'] else 'official metadata; abstract extraction unavailable'
    except Exception as e:
        row['abstract']=''; row['fetch_error']=str(e)
    return row
with ThreadPoolExecutor(max_workers=6) as pool:
    jobs=[pool.submit(enrich,r) for r in rows if r['candidate_id'] in selected]
    for f in as_completed(jobs):
        out.append(f.result())
        if len(out)%25==0: print(f'enriched {len(out)}/{len(jobs)}',flush=True)
out.sort(key=lambda r:r['candidate_id'])
(ROOT/'selected_enriched.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'total':len(out),'with_abstract':sum(bool(r.get('abstract')) for r in out),'errors':[(r['candidate_id'],r.get('fetch_error')) for r in out if r.get('fetch_error')]},ensure_ascii=False),flush=True)
