from pathlib import Path
from collections import defaultdict, Counter
import re,json,sys,unicodedata,html,csv,copy

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'v2_300篇'
ARCHIVE=ROOT.parent/'参考文献全文_300篇'
OUTNAME='01_视觉单目标跟踪模板更新与记忆机制综述_v3_期刊格式修订'
ZH={
4:('Survey of visual object tracking methods based on Transformer',['Sun Z W','Qian L Z','Yang C D','Gao Y B','Lu Q Y','Yuan G L'],['孙子文','钱立志','杨传栋','高一博','陆庆阳','袁广林'],'Journal of Computer Applications'),
5:('Visual intelligence evaluation techniques for single object tracking: a survey',['Hu S Y','Zhao X','Huang K Q'],['胡世宇','赵鑫','黄凯奇'],'Journal of Image and Graphics'),
47:('Multi-modal visual tracking: a survey',['Li C L','Lu A D','Liu L','Tang J'],['李成龙','鹿安东','刘磊','汤进'],'Journal of Image and Graphics'),
48:('Survey on visual single object tracking based on deep learning',['Zhang C G','Yang H T','Wang J Y','Feng B D','Li G Y','Gao Y G'],['张长弓','杨海涛','王晋宇','冯博迪','李高源','高宇歌'],'Application Research of Computers'),
49:('Context-aware attention fused Transformer tracking',['Xu H','Dong S H','Zhang J W','Zheng Y H'],['徐晗','董仕豪','张家伟','郑钰辉'],'Journal of Image and Graphics')}

def norm(s):return ''.join(c for c in unicodedata.normalize('NFKD',s).casefold() if c.isalnum())
def latex_clean(s):
    s=re.sub(r'\\[\"\'`^~=.uvHckbdtr]\s*\{?([A-Za-z])\}?',r'\1',s)
    return s.replace('{','').replace('}','').replace('\\&','&').strip()
def field(bib,name):
    m=re.search(r'\b'+name+r'\s*=\s*\{',bib,re.I)
    if not m:return ''
    i=m.end();start=i;depth=1
    while i<len(bib) and depth:
        if bib[i]=='{':depth+=1
        elif bib[i]=='}':depth-=1
        i+=1
    return bib[start:i-1]
def person(name):
    name=latex_clean(name).strip().strip('.')
    if ',' in name:
        family,given=[x.strip() for x in name.split(',',1)]
    else:
        words=name.split()
        if len(words)==1:return {'family':name,'given':'','formatted':name}
        last_initials=all(len(w.replace('.',''))<=1 for w in words[1:])
        if last_initials:family,given=words[0],' '.join(words[1:])
        elif len(words)>2 and words[:2]==['Di','Stefano']:family,given='Di Stefano',' '.join(words[2:])
        else:
            start=len(words)-1
            while start>0 and words[start-1].lower() in ['van','von','de','der','den','di','da','del']:start-=1
            family,given=' '.join(words[start:]),' '.join(words[:start])
    initials=' '.join(w[0].upper() for w in re.findall(r'[^\W\d_]+',given,flags=re.U))
    return {'family':family,'given':given,'formatted':family+(' '+initials if initials else '')}

def build_library():
    rows=json.loads((OLD/'library_300.json').read_text(encoding='utf-8'))
    changes=[]
    for r in rows:
        n=r['number']; original_authors=r['authors']
        f=ROOT/'metadata_cache'/f'{n:03d}.json'
        cr=json.loads(f.read_text(encoding='utf-8')) if f.exists() else {}
        meta=cr.get('metadata') if cr.get('matched') else None
        if meta:
            r['metadata_source']='https://api.crossref.org/works/'+meta['DOI']
            if not r.get('doi') and meta.get('DOI'):r['doi']=meta['DOI'];changes.append({'number':n,'field':'doi','value':r['doi']})
            for dest,src in [('pages','page'),('volume','volume'),('issue','issue'),('publisher','publisher')]:
                if meta.get(src) and not r.get(dest):r[dest]=meta[src]
            if meta.get('event',{}).get('location'):r['location']=meta['event']['location']
            if meta.get('container-title'):r['publication_full']=(meta['container-title'])[0]
        raw=field(r.get('bibtex_original',''),'author')
        if n in ZH:
            title,authors,cn,venue=ZH[n]
            r['title_en']=title;r['authors_en']=authors;r['authors']=cn;r['venue_en']=venue
            people=[person(a) for a in authors]
        elif n==3:
            # Complete author list checked against the downloaded first page.
            people=[person(a) for a in ['Smeulders, Arnold W. M.','Chu, Dung M.','Cucchiara, Rita','Calderara, Simone','Dehghan, Afshin','Shah, Mubarak']]
            r['author_source']='已下载作者稿首页，编号003'
        elif raw:
            people=[person(a) for a in re.split(r'\s+and\s+',raw) if a.strip()]
        elif meta and meta.get('author'):
            people=[]
            for a in meta['author']:
                if not a.get('family'):continue
                pp=person(a['family']+', '+a['given']) if a.get('given') else person(a['family']);people.append(pp)
        else:
            people=[person(a) for a in r['authors'] if a.lower().strip('. ') not in ['et al','等']]
            if any(a.lower().strip('. ') in ['et al','等'] for a in r['authors']):r['author_list_incomplete']=True
        r['people']=people
        r['authors_normalized']=[p['formatted'] for p in people]
        if raw and any(' and ' in a for a in original_authors):changes.append({'number':n,'field':'authors','from':original_authors,'to':r['authors_normalized']})
        r['citation_first']=r['authors'][0] if n in ZH else people[0]['family']
        r['citation_names']=(r['citation_first'] if len(people)==1 else r['citation_first']+'和'+(r['authors'][1] if n in ZH else people[1]['family']) if len(people)==2 else r['citation_first']+'等')
        r['sort_key']=' '.join(r['authors_normalized']).casefold()
    # Stable disambiguation: all references with the same displayed first author/year receive suffixes.
    groups=defaultdict(list)
    for r in rows:groups[(r['citation_names'],r['year'])].append(r)
    for g in groups.values():
        for i,r in enumerate(sorted(g,key=lambda r:(r['sort_key'],norm(r.get('title_en',r['title']))))):
            r['year_label']=str(r['year'])+(chr(97+i) if len(g)>1 else '')
            r['citation']=r['citation_names']+'，'+r['year_label']
    rows_sorted=sorted(rows,key=lambda r:(norm(r['sort_key']),r['year'],r['year_label'],norm(r['title'])))
    for i,r in enumerate(rows_sorted,1):r['alphabetical_order']=i
    (ROOT/'library_v3.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT/'bibliography_changes.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2),encoding='utf-8')
    return rows,rows_sorted

def authors_text(r):
    a=r['authors_normalized'][:6]
    if len(r['authors_normalized'])>6 or r.get('author_list_incomplete'):return ', '.join(a)+', et al'
    return ', '.join(a[:-1])+' and '+a[-1] if len(a)>1 else a[0]

def reference(r):
    n=r['number'];author=authors_text(r);year=r['year_label'];title=r.get('title_en',r['title']).rstrip('.')
    official_booktitle=latex_clean(field(r.get('bibtex_original',''),'booktitle'))
    venue=r.get('venue_en',official_booktitle or r.get('publication_full',r['venue']))
    # Crossref sometimes supplies the book series rather than the conference volume.
    if norm(venue)=='lecturenotesincomputerscience' and 'ECCV' in r['venue']:
        venue='Computer Vision – ECCV '+str(r['year'])+(' Workshops' if 'workshop' in r['venue'].lower() else '')
    vol=r.get('volume','');issue=r.get('issue','');pages=str(r.get('pages','')).replace('–','-')
    if r['type']=='journal':
        pub=venue+((', '+str(vol)) if vol else '')+(('('+str(issue)+')') if issue else '')+((': '+pages) if pages else '')
        out=f'{author}. {year}. {title}. {pub}'
    else:
        venues={'CVPR':'Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition','ICCV':'Proceedings of the IEEE/CVF International Conference on Computer Vision','ECCV':'Computer Vision – ECCV','NeurIPS':'Advances in Neural Information Processing Systems','ICLR':'International Conference on Learning Representations'}
        venue=venues.get(venue,venue)
        pub=(r.get('location','')+': ' if r.get('location') else '')+r.get('publisher','')
        out=f'{author}. {year}. {title}//{venue}'
        if pub:out+='. '+pub
        if pages:out+=': '+pages
    if n in ZH:
        cn=', '.join(r['authors'][:6])+('，等' if len(r['authors'])>6 else '')
        out+=f" ({cn}. {year}. {r['title']}. {r['venue']}"+(f', {vol}' if vol else '')+(f'({issue})' if issue else '')+(f': {pages}' if pages else '')+')'
    out+='.'
    if r.get('doi'):out+=f" [DOI: {r['doi']}](https://doi.org/{r['doi']})"
    else:out+=f" [原文入口]({r['url']})"
    return out

TABLE_EN={
1:'Comparison with related surveys and the scope of this review',
2:'Conceptual distinctions among updating and memory mechanisms',
3:'Updated representations and decision mechanisms in representative methods',
4:'Historical representations and retrieval operations in representative methods',
5:'Common benchmarks and their relevance to updating and memory',
6:'Selected results reported in Table 2 of the HIPTrack paper (%)',
7:'Reference-frame counts and TrackingNet performance reported in Table 4 of STMTrack',
8:'Update-interval comparison reported in Table 3 of the HIPTrack supplementary material',
9:'Thematic organization and coverage of the 300 bibliographic records',
10:'State representations and analytical roles of additional representative methods',
11:'Sources of numerical evidence and boundaries of interpretation'}
FIG_EN={
1:'Unified loop of historical retrieval, current localization, and state updating',
2:'Conceptual comparison of templates, explicit spatial memory, and compressed temporal states',
3:'Representative research threads in updating and memory mechanisms',
4:'Replotted ablation results from the original papers ((a) Reference-frame count in STMTrack; (b) Update interval in HIPTrack)',
5:'Publication-year distribution of the 300 records selected for this review'}

def build_text(rows,ordered):
    lookup={r['number']:r for r in rows};text=next(OLD.glob('01_*.md')).read_text(encoding='utf-8')
    body,tail=text.split('## 参考文献',1);appendix='## 附录：'+tail.split('## 附录：',1)[1]
    # Keep revision notes out of the article's abstract and main text.
    body=re.sub(r'^>.*\n','',body,flags=re.M)
    body=body.replace('Template Updating and Memory Mechanisms in Visual Single-Object Tracking: A Survey','Template updating and memory mechanisms in visual single-object tracking: a survey')
    body=body.replace('本文对相关研究进行综述','对相关研究进行综述')
    abstract=(ROOT/'english_abstract.md').read_text(encoding='utf-8').strip()
    body=re.sub(r'(?<=## Abstract\n\n).*?(?=\n\nKeywords:)',abstract,body,flags=re.S)
    # Add explicit placeholders without inventing author identities or funding.
    engtitle='Template updating and memory mechanisms in visual single-object tracking: a survey'
    body=body.replace(engtitle, '作者：[待填写实际作者及顺序]\n\n单位：[待填写单位全称、城市和邮编]\n\n中图分类号：[待核定]　文献标识码：A\n\n'+engtitle+'\n\nAuthors and affiliations: [To be completed by the authors]',1)
    cited=set()
    def cite(m):
        numbers=[]
        for item in re.split('[,，]',m.group(1)):
            if re.search('[-–—]',item):
                a,b=map(int,re.split('[-–—]',item.strip()));numbers.extend(range(a,b+1))
            else:numbers.append(int(item))
        cited.update(numbers)
        return '（'+'；'.join(lookup[n]['citation'] for n in numbers)+'）'
    body=re.sub(r'\[([0-9]+(?:\s*[-–—,，]\s*[0-9]+)*)\]',cite,body)
    for r in rows:
        stem=r['citation_names'];year=r['year_label'];full='（'+r['citation']+'）'
        for st in [stem,stem.replace('等','等人')]:
            body=body.replace(st+full,st+'（'+year+'）')
            body=body.replace(st+'，'+str(r['year'])+full,st+'（'+year+'）')
    # The index table documents coverage; it is not a claim of identical mechanisms across each group.
    body=body.replace('文献索引 |','作者—年份索引 |')
    body=body.replace('本稿仍以','本综述仍以').replace('本版','本稿')
    # Add bilingual captions, retaining provenance notes.
    lines=[]
    for line in body.splitlines():
        line=line.replace('/v2_300篇/figures/','/v3_期刊格式修订_20261028/figures/')
        lines.append(line)
        m=re.match(r'\*\*表(\d+)\s',line)
        if m:lines.extend(['',f'**Table {m[1]} {TABLE_EN[int(m[1])]}**'])
        m=re.match(r'图(\d+)\s',line)
        if m:lines.extend(['',f'Fig. {m[1]} {FIG_EN[int(m[1])]}'])
    body='\n'.join(lines)
    appendix=appendix.split('## 附录：文献索引与精读状态')[0].strip()
    appendix=appendix.replace('本附录用于阅读核验，不计作新的实验结果。','本附录说明数值来源与证据边界，不计作新的实验结果。\n\n**表11 数值证据的原始出处与解释边界**\n\n**Table 11 '+TABLE_EN[11]+'**')
    authors='\n\n## 作者信息与基金项目（待填写）\n\n第一作者简介：[姓名、性别、职称或在读学位、主要研究方向、E-mail。]\n\n通信作者：[姓名、职称、主要研究方向、E-mail。]\n\n其他作者简介：[按实际作者顺序填写。]\n\n基金项目：[据实际资助填写中文名称、英文名称和项目编号；无资助时据实处理。]\n'
    text=body.strip()+'\n\n## 参考文献 References\n\n'+'\n\n'.join(reference(r) for r in ordered)+'\n\n'+appendix+authors
    (ROOT/(OUTNAME+'.md')).write_text(text,encoding='utf-8')
    mapping=[{'原编号':r['number'],'原题名':r['title'],'作者年份':r['citation'],'字母排序序号':r['alphabetical_order'],'DOI':r.get('doi',''),'作者信息待补':bool(r.get('author_list_incomplete')),'原始入口':r['url']} for r in rows]
    with (ROOT/'02_原编号与作者年份对照.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=mapping[0].keys());w.writeheader();w.writerows(mapping)
    stats={'references':len(rows),'cited_unique':len(cited),'uncited_numbers':sorted(set(lookup)-cited),'doi_present':sum(bool(r.get('doi')) for r in rows),'doi_missing':[r['number'] for r in rows if not r.get('doi')],'incomplete_author_lists':[r['number'] for r in rows if r.get('author_list_incomplete')],'english_abstract_words':len(re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*",abstract))}
    (ROOT/'content_validation.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(stats,ensure_ascii=False,indent=2))

if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');rows,ordered=build_library();build_text(rows,ordered)
