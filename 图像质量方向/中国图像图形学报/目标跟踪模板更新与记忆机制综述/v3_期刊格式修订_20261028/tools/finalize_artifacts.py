from pathlib import Path
import subprocess,sys,json,re,unicodedata,csv
import fitz
from docx import Document
from openpyxl import Workbook
from openpyxl.styles import Font,PatternFill,Alignment

ROOT=Path(__file__).resolve().parents[1]
NAME='01_视觉单目标跟踪模板更新与记忆机制综述_v3_期刊格式修订'
def norm(s):return ''.join(c for c in unicodedata.normalize('NFKD',s).casefold() if c.isalnum())

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    src=ROOT/(NAME+'.html');out=src.with_suffix('.pdf')
    args=[r'C:\Program Files (x86)\Microsoft\EdgeCore\155.0.4283.62\msedge.exe','--headless=new','--no-sandbox','--disable-gpu','--disable-extensions','--disable-background-networking','--no-first-run','--no-pdf-header-footer','--user-data-dir='+str(ROOT.parent.parent.parent/'pdf_render_v3_local'),'--print-to-pdf='+str(out),src.as_uri()]
    # A disposable profile renders only locally generated, script-free HTML and PNGs.
    result=subprocess.run(args,capture_output=True,timeout=50,creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode or not out.exists():raise RuntimeError('PDF renderer failed: '+str(result.returncode))
    pdf=fitz.open(out);alltext='\n'.join(p.get_text() for p in pdf)
    for i,page in enumerate(pdf):page.insert_text((285,822),str(i+1),fontsize=8,color=(.2,.2,.2))
    md=(ROOT/(NAME+'.md')).read_text(encoding='utf-8');toc=[];cursor=0
    for level,title in re.findall(r'^(#{1,3}) (.*)$',md,re.M):
        if len(level)==1:continue
        key=norm(title)
        for j in range(cursor,len(pdf)):
            if key in norm(pdf[j].get_text()):toc.append([len(level)-1,title,j+1]);cursor=j;break
    if toc:pdf.set_toc(toc)
    pdf.set_metadata({'title':'视觉单目标跟踪中的模板更新与记忆机制综述','subject':'期刊格式修订稿v3；同内容排版预览，作者信息及部分文献信息待核定','author':''})
    tmp=ROOT/'_numbered.pdf';pdf.save(tmp,garbage=4,deflate=True);pdf.close();tmp.replace(out)
    pdf=fitz.open(out);doc=Document(ROOT/(NAME+'.docx'))
    validation={'pdf_pages':len(pdf),'pdf_text_characters':len(alltext),'editable_equations':len(doc.element.xpath('.//m:oMath')),'tables':len(doc.tables),'sections_with_two_columns':sum('w:num="2"' in s._sectPr.xml for s in doc.sections),'bookmarks':len(toc),'numeric_citation_remnants':len(re.findall(r'\[\d+(?:[,–-]\d+)*\]',alltext))}
    previews={0:'preview_page1',2:'preview_body',len(pdf)-1:'preview_last'}
    for i,p in enumerate(pdf):
        t=p.get_text()
        for marker,name in [('Table 6 ','preview_results'),('Table 9 ','preview_index'),('参考文献 References','preview_references')]:
            if marker in t:previews[i]=name
    for i,n in previews.items():pdf[i].get_pixmap(matrix=fitz.Matrix(1.3,1.3)).save(ROOT/(n+'.png'))
    (ROOT/'final_validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding='utf-8')
    library=json.loads((ROOT/'library_v3.json').read_text(encoding='utf-8'))
    wb=Workbook();ws=wb.active;ws.title='原编号与新版引用';ws.append(['原编号','题名','新版作者年份','参考文献排序','DOI','出版出处','页码','作者完整列表','原文入口'])
    for r in library:
        ws.append([r['number'],r['title'],r['citation'],r['alphabetical_order'],r.get('doi',''),r['venue'],r.get('pages',''),'; '.join(r['authors_normalized']),r['url']])
    ws.freeze_panes='C2';ws.auto_filter.ref=ws.dimensions
    for c in ws[1]:c.fill=PatternFill('solid',fgColor='25495C');c.font=Font(color='FFFFFF',bold=True)
    for k,v in {'A':8,'B':65,'C':30,'D':15,'E':36,'F':45,'G':18,'H':70,'I':50}.items():ws.column_dimensions[k].width=v
    miss=wb.create_sheet('DOI待核定');miss.append(['原编号','题名','年份','原文入口'])
    for r in library:
        if not r.get('doi'):miss.append([r['number'],r['title'],r['year'],r['url']])
    for k,v in {'A':8,'B':75,'C':10,'D':90}.items():miss.column_dimensions[k].width=v
    wb.save(ROOT/'02_引用对照及待核验清单.xlsx')
    bib=[];ris=[]
    for r in library:
        au=' and '.join(p['family']+', '+p['given'] for p in r['people']);fields={'author':au,'title':r.get('title_en',r['title']),'year':r['year'],'journal' if r['type']=='journal' else 'booktitle':r.get('venue_en',r['venue']),'doi':r.get('doi',''),'pages':r.get('pages',''),'url':r['url']}
        bib.append('@'+('article' if r['type']=='journal' else 'inproceedings')+'{ref'+str(r['number']).zfill(3)+',\n'+',\n'.join(f'  {k} = {{{v}}}' for k,v in fields.items() if v)+'\n}')
        lines=['TY  - '+('JOUR' if r['type']=='journal' else 'CONF'),f'ID  - ref{r["number"]:03d}',f'TI  - {r.get("title_en",r["title"])}',f'PY  - {r["year"]}',f'T2  - {r.get("venue_en",r["venue"])}']
        lines += ['AU  - '+p['family']+', '+p['given'] for p in r['people']]
        if r.get('doi'):lines.append('DO  - '+r['doi'])
        lines+=['UR  - '+r['url'],'ER  -'];ris.append('\n'.join(lines))
    (ROOT/'03_300篇参考文献_补查版.bib').write_text('\n\n'.join(bib),encoding='utf-8')
    (ROOT/'03_300篇参考文献_补查版.ris').write_text('\n\n'.join(ris),encoding='utf-8')
    missing=sum(not r.get('doi') for r in library)
    readme=f'''# v3期刊格式修订稿使用说明

修订日期：2026年10月28日。基于v2／300篇文献阅读稿，参照《中国图象图形学报》2026版模板修订。v1、v2均保留。

正文文件为 `01_视觉单目标跟踪模板更新与记忆机制综述_v3_期刊格式修订.docx`。PDF为同内容的排版预览（{len(pdf)}页），由HTML生成，不是Word排版引擎直接导出，二者分页可能不同。正式送审应在最终填写作者信息后，用实际使用的Word软件再核对分页。

已完成：

- 英文摘要扩展为1032词，按Objective、Method、Result、Conclusion组织；中文综述摘要保留连续段落。
- 保留300条文献，全文数字编号引用转换为作者—年份；同组同年引文加字母区分；文后按作者字母排序。
- DOI从52条补至{300-missing}条；修复官方BibTeX导入时的部分作者拆分错误；补齐5条中文文献的英文对照。
- 正文采用双栏结构，图表跨栏；5幅主图提供400 dpi PNG及SVG；11张表改为三线表并补英文题名。
- 6个公式改为Word原生可编辑公式，连续编号；PDF中公式为高分辨率渲染结果。
- 提供原编号—作者年份对照表、BibTeX和RIS，方便与已下载PDF逐条核验。

仍需核定：

1. 作者、顺序、单位、基金、简介与中图分类号均保留明确待填位置，未代填导师或项目。
2. {missing}条文献的DOI未取得可靠匹配，已保留原文入口并列入Excel“DOI待核定”工作表；空白不表示该论文没有DOI。
3. 文献下载、书目匹配和作者姓名校正不等于全文精读。方法解释、比较论点与原文页码仍需按下载清单逐条核验。表9保留全部文献的主题索引，不能代替每篇论文的深入讨论。
4. 没有新增复现实验；表6—表8及图4仍是来源明确的已发表结果，未改写成原创实验。作者稿、预印本和正式出版版的表号或页码需分别对应。
5. 本版是可继续修改的期刊格式修订稿，未作为“已满足全部投稿条件”的最终稿，也未提交期刊。

参考文献全文与下载状态位于同级 `参考文献全文_300篇` 文件夹。PDF文件名仍按v2原编号命名；本目录对照表给出v3引用。
'''
    (ROOT/'00_修改说明与待核定事项.md').write_text(readme,encoding='utf-8')
    print(json.dumps(validation,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
