from pathlib import Path
import re,json,sys
import fitz
sys.stdout.reconfigure(encoding='utf-8')
root=Path(__file__).resolve().parent
stem='01_综述完整初稿_v2_300篇文献_20261025'
md=(root/(stem+'.md')).read_text(encoding='utf-8')
expected=re.findall(r'(?<!!)\[([^\]]+)\]\(([^)]+)\)',md)
clean=lambda s:re.sub(r'\s+','',s)
with fitz.open(root/(stem+'.pdf')) as pdf:
    spans=[]
    for i,page in enumerate(pdf):
        for b in page.get_text('dict')['blocks']:
            for l in b.get('lines',[]):
                for s in l['spans']:
                    if s['color']==255 and clean(s['text']):spans.append((i,s))
    found=''.join(clean(s['text']) for _,s in spans)
    wanted=''.join(clean(label) for label,url in expected)
    assert found==wanted,{'found_length':len(found),'expected_length':len(wanted),'blue_spans':len(spans),'expected_links':len(expected),'sample':found[:160]}
    # Every source label may wrap across several spans or pages.
    pending=[];ix=0
    for label,url in expected:
        remainder=clean(label)
        while remainder:
            page,s=spans[ix];value=clean(s['text'])
            assert remainder.startswith(value),(label,value,remainder)
            pending.append((page,fitz.Rect(s['bbox']),url))
            remainder=remainder[len(value):];ix+=1
    assert ix==len(spans)
    # Allow repeat execution without duplicate annotations.
    for page in pdf:
        for link in page.get_links():page.delete_link(link)
    for p,rect,url in pending:pdf[p].insert_link({'kind':fitz.LINK_URI,'from':rect,'uri':url})
    pdf.saveIncr()
    print(json.dumps({'linked_labels':len(expected),'link_annotations':len(pending)},ensure_ascii=False))
