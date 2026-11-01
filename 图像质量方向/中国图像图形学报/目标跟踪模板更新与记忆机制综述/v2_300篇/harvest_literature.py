from pathlib import Path
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import requests, re, json, gzip, hashlib, time, sys
from bs4 import BeautifulSoup

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parent
CACHE = ROOT / 'source_cache'
CACHE.mkdir(exist_ok=True)
STAMP = '2026-10-25'

def fetch(url):
    name = hashlib.sha256(url.encode()).hexdigest()[:24]
    dest = CACHE / (name + '.html.gz')
    if dest.exists():
        return gzip.decompress(dest.read_bytes()).decode('utf-8')
    errors = []
    for attempt in range(2):
        try:
            r = requests.get(url, timeout=(8, 35), headers={'User-Agent': 'ResearchBibliography/1.0'})
            r.raise_for_status()
            content = r.content.decode('utf-8', errors='replace')
            dest.write_bytes(gzip.compress(content.encode('utf-8')))
            (CACHE / (name + '.json')).write_text(json.dumps({'url': url, 'access_date': STAMP, 'sha256': hashlib.sha256(r.content).hexdigest()}, indent=2), encoding='utf-8')
            return content
        except Exception as e:
            errors.append(str(e))
    raise RuntimeError(' | '.join(errors))

def bibfield(bib, name):
    m = re.search(r'\b' + re.escape(name) + r'\s*=\s*\{', bib, re.I)
    if not m:
        return ''
    level = 1
    i = m.end()
    j = i
    while j < len(bib):
        if bib[j] == '{': level += 1
        elif bib[j] == '}': level -= 1
        if not level: return bib[i:j]
        j += 1
    return ''

def norm(title):
    return re.sub(r'[^a-z0-9]', '', title.lower())

def parse_index(url):
    soup = BeautifulSoup(fetch(url), 'lxml')
    records = []
    if not soup.select('dt.ptitle'):
        daily = sorted({urljoin(url, a['href']) for a in soup.find_all('a', href=True)
                        if re.search(r'day=20\d{2}-\d{2}-\d{2}', a['href'])})
        for child in daily:
            records.extend(parse_index(child))
        return records
    for dt in soup.select('dt.ptitle'):
        anchor = dt.find('a')
        if not anchor: continue
        title = ' '.join(anchor.get_text(' ', strip=True).split()).strip('"')
        # Broad discovery pool; final inclusion is reviewed separately.
        if not re.search(r'track|video object segmentation|space.time memory|video.*memory|memory.*video', title, re.I):
            continue
        landing = urljoin(url, anchor['href'])
        siblings = []
        for el in dt.next_siblings:
            if getattr(el, 'name', None) == 'dt': break
            if getattr(el, 'name', None) == 'dd': siblings.append(el)
        if len(siblings) < 2: continue
        authors = [x.get('value', '') for x in siblings[0].select('input[name=query_author]')]
        if not authors:
            authors = [x.strip().rstrip('*').strip() for x in siblings[0].get_text(' ', strip=True).split(',')]
        links = [(a.get_text(' ', strip=True), urljoin(url, a['href'])) for dd in siblings for a in dd.find_all('a', href=True)]
        pdf = next((link for label, link in links if label.lower() == 'pdf'), '')
        doi = next((link.split('doi.org/')[-1] if 'doi.org/' in link else link.split('/chapter/')[-1] for label, link in links if label.lower() == 'doi'), '')
        bib = next((b.get_text('\n', strip=True) for dd in siblings for b in dd.select('.bibref')), '')
        ym = re.search(r'(CVPR|ICCV|WACV|ECCV)[_/]?(20\d{2})', landing, re.I)
        if not ym: ym = re.search(r'(eccv)_(20\d{2})', landing, re.I)
        venue = ym[1].upper() if ym else ''
        year = int(ym[2]) if ym else int(bibfield(bib, 'year') or 0)
        records.append({'title': title, 'authors': authors, 'venue': venue, 'year': year,
                        'pages': bibfield(bib, 'pages'), 'doi': doi, 'url': landing, 'pdf': pdf,
                        'bibtex_original': bib, 'source_index': url, 'access_date': STAMP,
                        'verification': 'official proceedings metadata', 'title_key': norm(title)})
    return records

if __name__ == '__main__':
    sources = [f'https://openaccess.thecvf.com/CVPR{y}?day=all' for y in range(2013, 2027)]
    sources += [f'https://openaccess.thecvf.com/ICCV{y}?day=all' for y in range(2013, 2026, 2)]
    sources = [u.split('?')[0] if any(v in u for v in ['CVPR2018?', 'CVPR2019?', 'CVPR2020?', 'ICCV2019?']) else u for u in sources]
    sources += ['https://www.ecva.net/papers.php']
    gathered, errors = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {pool.submit(parse_index, u): u for u in sources}
        for f in as_completed(jobs):
            u = jobs[f]
            try:
                rows = f.result(); gathered.extend(rows)
                print(json.dumps({'source': u, 'candidates': len(rows)}, ensure_ascii=False), flush=True)
            except Exception as e:
                errors.append({'url': u, 'error': str(e)})
                print(json.dumps(errors[-1], ensure_ascii=False), flush=True)
    unique = {}
    for r in gathered: unique.setdefault(r['title_key'], r)
    rows = sorted(unique.values(), key=lambda r: (r['year'], r['venue'], r['title'].lower()))
    for i, r in enumerate(rows, 1): r['candidate_id'] = i
    (ROOT / 'candidates.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    (ROOT / 'discovery_errors.json').write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding='utf-8')
    (ROOT / 'candidate_titles.tsv').write_text('\n'.join(f"{r['candidate_id']}\t{r['year']}\t{r['venue']}\t{r['title']}" for r in rows), encoding='utf-8')
    print(json.dumps({'total_unique': len(rows), 'by_venue': dict(Counter(r['venue'] for r in rows))}, ensure_ascii=False), flush=True)
