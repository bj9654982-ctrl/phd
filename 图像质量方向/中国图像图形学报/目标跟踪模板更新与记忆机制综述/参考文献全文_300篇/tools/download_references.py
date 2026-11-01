"""Download public primary-source PDFs, preserving a resumable per-reference audit."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin, urlsplit, parse_qs
from bs4 import BeautifulSoup
import requests, json, re, time, hashlib, threading, sys, argparse
import fitz

ROOT=Path(__file__).resolve().parents[1]
LIB=ROOT.parent/'v2_300篇'/'library_300.json'
PDF=ROOT/'PDF'; META=ROOT/'records'; CACHE=ROOT/'source_pages'
for p in (PDF,META,CACHE):p.mkdir(parents=True,exist_ok=True)
local=threading.local()
host_locks={}; lock=threading.Lock()

def session():
    if not hasattr(local,'session'):
        local.session=requests.Session()
        local.session.headers['User-Agent']='Mozilla/5.0 (compatible; LiteratureArchive/1.0)'
    return local.session

def read_url(url):
    host=urlsplit(url).netloc
    with lock:sem=host_locks.setdefault(host,threading.Semaphore(4))
    with sem:
        r=session().get(url,timeout=(15,75),stream=True)
        r.raise_for_status()
        chunks=[]; size=0
        for chunk in r.iter_content(262144):
            chunks.append(chunk);size+=len(chunk)
            if size>150*1024*1024:
                r.close();raise ValueError('File exceeds 150 MB; manual review needed')
        return b''.join(chunks),r.url,r.headers.get('Content-Type','')

def filename(row):
    title=re.sub(r'[<>:"/\\|?*\x00-\x1f]','_',row['title'])
    title=re.sub(r'\s+','_',title).strip(' ._')[:78]
    return f"{int(row['number']):03d}_{row['year']}_{title}.pdf"

def normalize(t):
    return ''.join(c.casefold() for c in t if c.isalnum())

def get_links(data,base):
    soup=BeautifulSoup(data,'lxml');out=[]
    for m in soup.find_all('meta'):
        if (m.get('name','').lower() in ['citation_pdf_url','eprints.document_url']) and m.get('content'):
            out.append(urljoin(base,m['content']))
    candidates=[]
    for a in soup.find_all('a',href=True):
        href=urljoin(base,a['href']);text=a.get_text(' ',strip=True).lower()
        if re.search(r'supplement|appendix|poster|slide',href+' '+text,re.I):continue
        score=0
        if re.search(r'\.pdf(?:[?#]|$)',href,re.I):score+=5
        if text.endswith('.pdf'):score+=6
        if text in ['pdf','full text','download pdf','全文','pdf下载','全文下载']:score+=5
        if 'lowqualitypdf' in href:score+=6
        if '/article/download/' in href:score+=5
        if score:candidates.append((score,href))
    for tag in soup.find_all(['iframe','embed']):
        src=tag.get('src','')
        if '.pdf' in src or '/article/download/' in src:out.append(urljoin(base,src))
    out.extend(href for score,href in sorted(candidates,reverse=True))
    return list(dict.fromkeys(out))[:6]

def initial_urls(row,overrides):
    out=[]
    for u in overrides.get(str(row['number']),[]):out.append(u['url'] if isinstance(u,dict) else u)
    if row.get('pdf'):out.append(row['pdf'])
    for u in [row['url']]+(row.get('all_source_urls') or []):
        if 'supplement' in u.lower():continue
        if 'arxiv.org/abs/' in u:out.append(u.replace('/abs/','/pdf/'))
        elif 'openreview.net/forum?' in u:
            ident=parse_qs(urlsplit(u).query).get('id',[''])[0]
            out.append('https://openreview.net/pdf?id='+ident)
        elif '.pdf' in u:out.append(u)
        out.append(u)
    return list(dict.fromkeys(out))

def version(url):
    host=urlsplit(url).netloc
    if 'arxiv.org' in host:return 'arXiv公开版本；可能与正式出版版不同'
    if any(x in host for x in ['thecvf.com','ecva.net','papers.nips.cc','bmva.org']):return '会议官方开放版本'
    if 'openreview.net' in host:return 'OpenReview公开会议版本'
    if any(x in host for x in ['cjig.cn','joca.cn','arocmag.cn','aaai.org','springer','sciencedirect','ieee.org','link.springer']):return '出版方公开全文'
    return '作者或机构公开稿；待人工确认版本'

def download(row,overrides):
    result={'number':row['number'],'title':row['title'],'year':row['year'],'authors':row['authors'],
      'venue':row['venue'],'category':row['category'],'priority':row['priority'],
      'original_source':row['url'],'doi':row.get('doi',''),'status':'not_downloaded',
      'attempts':[],'access_date':'2026-10-28','human_verification':'未核验'}
    queue=[(u,0) for u in initial_urls(row,overrides)];seen=set()
    while queue and len(seen)<16:
        url,depth=queue.pop(0)
        if url in seen or not url.startswith(('http://','https://')):continue
        seen.add(url)
        try:
            data,final,content_type=read_url(url)
            attempt={'url':url,'final_url':final,'bytes':len(data),'content_type':content_type}
            result['attempts'].append(attempt)
            if b'%PDF-' in data[:1024]:
                path=PDF/filename(row);path.write_bytes(data)
                result.update(status='downloaded',file=str(path.relative_to(ROOT)),
                  download_url=final,requested_url=url,bytes=len(data),
                  sha256=hashlib.sha256(data).hexdigest(),version=version(final))
                return result
            attempt['result']='not_pdf'
            if len(data)<12*1024*1024 and depth<2:
                (CACHE/f"{row['number']:03d}_{hashlib.sha256(url.encode()).hexdigest()[:12]}.html").write_bytes(data)
                links=get_links(data,final)
                queue[0:0]=[(u,depth+1) for u in links if u not in seen]
        except Exception as e:
            result['attempts'].append({'url':url,'error':f'{type(e).__name__}: {e}'[:650]})
    return result

def validate(result):
    if result['status']!='downloaded':return result
    path=ROOT/result['file']
    try:
        with fitz.open(path) as d:
            result['pages']=len(d);result['pdf_title']=d.metadata.get('title','')
            text='\n'.join(d[i].get_text() for i in range(min(2,len(d))))
            result['first_page_text']=d[0].get_text()[:14000] if len(d) else ''
        target=normalize(result['title']);ntext=normalize(text)
        exact=target in ntext
        tokens=set(re.findall(r'[a-zA-Z]{3,}',result['title'].lower()))
        words=set(re.findall(r'[a-zA-Z]{3,}',text.lower()))
        coverage=round(len(tokens & words)/len(tokens),3) if tokens else None
        result['title_exact_normalized']=exact;result['title_token_coverage']=coverage
        result['identity_check']='题名自动匹配；待人工核验' if exact else '题名需人工核对'
        result['technical_check']='PDF可解析'
    except Exception as e:
        result['status']='invalid_pdf';result['technical_check']=str(e)
    return result

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap=argparse.ArgumentParser();ap.add_argument('--retry',action='store_true');ap.add_argument('--numbers',default='');args=ap.parse_args()
    library=json.loads(LIB.read_text(encoding='utf-8'))
    override_file=ROOT/'source_overrides.json'
    overrides=json.loads(override_file.read_text(encoding='utf-8')) if override_file.exists() else {}
    selected=set(map(int,args.numbers.split(','))) if args.numbers else set()
    pending=[]
    for row in library:
        if selected and row['number'] not in selected:continue
        f=META/f"{row['number']:03d}.json"
        if f.exists():
            old=json.loads(f.read_text(encoding='utf-8'))
            if old.get('status')=='downloaded' and (ROOT/old['file']).exists():continue
            if not args.retry:continue
        pending.append(row)
    print(f'Pending: {len(pending)}',flush=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures={pool.submit(download,r,overrides):r for r in pending}
        done=0
        for future in as_completed(futures):
            result=validate(future.result());done+=1
            f=META/f"{result['number']:03d}.json"
            f.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
            if done%10==0 or result['status']!='downloaded':
                print(f"{done}/{len(pending)} [{result['number']:03d}] {result['status']} {result.get('pages','')} pages",flush=True)
    rows=[json.loads(f.read_text(encoding='utf-8')) for f in sorted(META.glob('*.json'))]
    print('Completed records:',len(rows),'Downloaded:',sum(r['status']=='downloaded' for r in rows),flush=True)
    print('Unresolved:',[(r['number'],r['title']) for r in rows if r['status']!='downloaded'],flush=True)
    print('Title needs checking:',[(r['number'],r.get('title_token_coverage')) for r in rows if r.get('status')=='downloaded' and not r.get('title_exact_normalized')],flush=True)

if __name__=='__main__':main()
