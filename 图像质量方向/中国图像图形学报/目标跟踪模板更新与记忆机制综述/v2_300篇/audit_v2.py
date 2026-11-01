from pathlib import Path
from collections import Counter
import re,json,sys,zipfile
import fitz
from docx import Document
from openpyxl import load_workbook

sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parent
STEM='01_综述完整初稿_v2_300篇文献_20261025'
text=(ROOT/(STEM+'.md')).read_text(encoding='utf-8')
body=text.split('## 参考文献')[0]
refs={int(n) for n in re.findall(r'^\[(\d+)\]',text,re.M)}
used=set()
for expr in re.findall(r'\[([0-9,–-]+)\]',body):
    for seg in expr.split(','):
        if '–' in seg or '-' in seg:
            lo,hi=map(int,re.split('[–-]',seg));used.update(range(lo,hi+1))
        else:used.add(int(seg))
records=json.loads((ROOT/'library_300.json').read_text(encoding='utf-8'))
doc=Document(ROOT/(STEM+'.docx'))
wb=load_workbook(ROOT/'02_300篇文献分类索引.xlsx',read_only=True)
with fitz.open(ROOT/(STEM+'.pdf')) as pdf:
    alltext='\n'.join(p.get_text() for p in pdf)
    anchors=[]
    for i,p in enumerate(pdf):
        for b in p.get_text('dict')['blocks']:
            for line in b.get('lines',[]):
                for s in line['spans']:
                    if '原始来源' in s['text']:
                        anchors.append({'page':i+1,'text':s['text'],'color':s['color'],'bbox':s['bbox']})
    for needle,name in [('表9 本稿','preview_coverage.png'),('表10 新增','preview_comparison.png'),('[36]','preview_references.png'),('[300]','preview_last.png')]:
        for p in pdf:
            if needle in p.get_text():
                p.get_pixmap(matrix=fitz.Matrix(1.2,1.2)).save(ROOT/name);break
    bib=(ROOT/'03_300篇参考文献.bib').read_text(encoding='utf-8')
    ris=(ROOT/'03_300篇参考文献.ris').read_text(encoding='utf-8')
    result={
        'bibliography_entries':len(refs),'body_cited_entries':len(used),'missing_reference_numbers':sorted(used-refs),'uncited_entries':sorted(refs-used),
        'unresolved_placeholders':re.findall(r'\[@[^\]]*\]|<!-- INSERT:|【待|TODO',text),
        'image_paths_exist':all(Path(p).exists() for p in re.findall(r'^!\[.*?\]\((.*?)\)',text,re.M)),
        'docx_tables':len(doc.tables),'docx_images':len(doc.inline_shapes),'pdf_pages':len(pdf),'pdf_bookmarks':len(pdf.get_toc()),
        'pdf_final_reference_present':'[300]' in alltext,'pdf_end_present':'原论文报告值未由本稿重新验证' in alltext,
        'bibtex_entries':len(re.findall(r'^@(?:article|inproceedings)\{',bib,re.M)),
        'ris_entries':len(re.findall(r'^TY  - ',ris,re.M)),
        'xlsx_rows_all':wb['全部300篇'].max_row-1,'xlsx_priority_rows':wb['优先精读50篇'].max_row-1,
        'records_without_authors':[r['key'] for r in records if not r.get('authors')],
        'records_without_source':[r['key'] for r in records if not r.get('url','').startswith('http')],
        'year_distribution':dict(sorted(Counter(r['year'] for r in records).items())),
        'pdf_link_count':sum(len(p.get_links()) for p in pdf),
        'source_anchor_spans':len(anchors),'anchor_color_counts':dict(Counter(s['color'] for s in anchors)),
    }
    (ROOT/'pdf_source_anchor_positions.json').write_text(json.dumps(anchors,ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT/'final_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    assert refs==set(range(1,301)) and used==refs
    assert not result['unresolved_placeholders']
    assert result['image_paths_exist'] and result['pdf_final_reference_present'] and result['pdf_end_present']
    assert result['bibtex_entries']==result['ris_entries']==result['xlsx_rows_all']==300
    assert result['xlsx_priority_rows']==50
    assert not result['records_without_authors'] and not result['records_without_source']
