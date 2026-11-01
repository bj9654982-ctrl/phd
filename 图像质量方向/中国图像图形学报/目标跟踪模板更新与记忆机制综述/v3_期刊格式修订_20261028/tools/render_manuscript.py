from pathlib import Path
import re,json,sys,html,shutil
from docx import Document
from docx.shared import Pt,Cm,Inches
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.section import WD_SECTION_START
from docx.enum.text import WD_ALIGN_PARAGRAPH,WD_BREAK,WD_TAB_ALIGNMENT
from docx.opc.constants import RELATIONSHIP_TYPE as RT
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from collections import Counter

ROOT=Path(__file__).resolve().parents[1];OLD=ROOT.parent/'v2_300篇'
NAME='01_视觉单目标跟踪模板更新与记忆机制综述_v3_期刊格式修订'
FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)
sys.stdout.reconfigure(encoding='utf-8')

def generate_figures():
    # Reuse the existing, reproducible figure definitions with a higher raster resolution.
    source=(OLD/'render_v2.py').read_text(encoding='utf-8').split('text=SRC.read_text',1)[0]
    source=source.replace('ROOT=Path(__file__).resolve().parent',f'ROOT=Path({str(ROOT)!r})').replace('dpi=220','dpi=400')
    exec(compile(source,'figure_definitions','exec'),{'__file__':str(ROOT/'_figure_definitions.py')})
    lib=json.loads((ROOT/'library_v3.json').read_text(encoding='utf-8'));counts=Counter(r['year'] for r in lib)
    plt.rcParams['font.family']='Microsoft YaHei'
    f,ax=plt.subplots(figsize=(10.5,3.4));years=list(range(min(counts),max(counts)+1))
    ax.bar(years,[counts[y] for y in years],color='#46697B',width=.76)
    ax.set_xlabel('发表年份');ax.set_ylabel('选入文献数量 / 篇');ax.set_xticks(years[::2]);ax.tick_params(axis='x',rotation=45)
    ax.grid(axis='y',alpha=.15);ax.spines[['top','right']].set_visible(False)
    f.tight_layout();f.savefig(FIG/'fig5_years.png',dpi=400);f.savefig(FIG/'fig5_years.svg');plt.close(f)

def parse(text):
    lines=text.splitlines();i=0;blocks=[]
    while i<len(lines):
        s=lines[i].strip()
        if not s:i+=1;continue
        if s==r'\[':
            i+=1;eq=[]
            while i<len(lines) and lines[i].strip()!=r'\]':eq.append(lines[i].strip());i+=1
            blocks.append(('math',' '.join(eq)));i+=1;continue
        if s.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                cells=[x.strip() for x in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[-: ]+',x) for x in cells):rows.append(cells)
                i+=1
            blocks.append(('table',rows));continue
        m=re.match(r'^(#{1,3}) (.*)',s)
        if m:blocks.append(('heading',(len(m[1]),m[2])));i+=1;continue
        m=re.fullmatch(r'!\[(.*?)\]\((.*?)\)',s)
        if m:blocks.append(('image',(m[1],m[2])));i+=1;continue
        blocks.append(('paragraph',s));i+=1
    return blocks

def el(tag,children=(),value=None):
    node=OxmlElement('m:'+tag)
    if value is not None:node.set(qn('m:val'),value)
    for x in children:node.append(x)
    return node
def mt(t,style='i'):
    r=el('r');r.append(el('rPr',[el('sty',value=style)]));p=OxmlElement('w:rPr');fonts=OxmlElement('w:rFonts');fonts.set(qn('w:ascii'),'Cambria Math');fonts.set(qn('w:hAnsi'),'Cambria Math');p.append(fonts);size=OxmlElement('w:sz');size.set(qn('w:val'),'21');p.append(size);r.append(p)
    tx=el('t');tx.text=t;r.append(tx);return r
def seq(*x):return [mt(a,'p') if isinstance(a,str) else a for a in x]
def sub(s,i,bold=True):return el('sSub',[el('e',[mt(s,'bi' if bold else 'i')]),el('sub',[mt(i)])])
def hatb():return el('sSub',[el('e',[el('acc',[el('accPr',[el('chr',value='̂')]),el('e',[mt('b','bi')])])]),el('sub',[mt('t')])])
def tildez():return el('sSub',[el('e',[el('acc',[el('accPr',[el('chr',value='̃')]),el('e',[mt('z','bi')])])]),el('sub',[mt('t')])])
def sumterm():return el('nary',[el('naryPr',[el('chr',value='∑'),el('limLoc',value='undOvr')]),el('sub',seq(mt('i'),'=',mt('1','p'))),el('sup',[mt('K')]),el('e',seq(sub('a','t,i',False),sub('v','i')))])
def math_nodes(n):
    s=lambda i='t':sub('s',i);r=lambda:sub('r','t');M=lambda i:sub('M',i);q=lambda:sub('q','t',False)
    if n==1:return seq(s(),' = S(',sub('I','t'),', ',sub('b̂','t−1'),'),    ',r(),' = R(',s(),', ',M('t−1'),')')
    if n==2:return seq(hatb(),' = L(',s(),', ',r(),'),    ',sub('c','t'),' = E(',sub('I','t'),', ',hatb(),'),    ',q(),' = Q(',s(),', ',r(),', ',hatb(),')')
    if n==3:return seq(M('t'),' = U(',M('t−1'),', ',sub('c','t'),', ',q(),')')
    if n==4:return seq(sub('z','t'),' = (1 − ',sub('η','t',False),')',sub('z','t−1'),' + ',sub('η','t',False),tildez(),',    0 ≤ ',sub('η','t',False),' ≤ 1')
    if n==5:
        second=el('nary',[el('naryPr',[el('chr',value='∑'),el('limLoc',value='undOvr')]),el('sub',seq(mt('i'),'=1')),el('sup',[mt('K')]),el('e',[sub('a','t,i',False)])])
        return seq(r(),' = ',sumterm(),',    ',second,' = 1')
    frac=el('f',[el('num',seq('|',hatb(),' ∩ ',sub('b','t'),'|')),el('den',seq('|',hatb(),' ∪ ',sub('b','t'),'|'))])
    return seq(sub('o','t',False),' = ',frac)

def run_font(run,size=None,bold=None):
    run.font.name='Times New Roman';run._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'宋体')
    if size:run.font.size=Pt(size)
    if bold is not None:run.bold=bold
def inline(p,text,size=None):
    for t in re.split(r'(\*\*.*?\*\*|\[[^\]]+\]\([^)]+\))',text):
        if not t:continue
        m=re.fullmatch(r'\[([^\]]+)\]\(([^)]+)\)',t)
        if m:
            h=OxmlElement('w:hyperlink');h.set(qn('r:id'),p.part.relate_to(m[2],RT.HYPERLINK,is_external=True))
            r=OxmlElement('w:r');props=OxmlElement('w:rPr');fonts=OxmlElement('w:rFonts');fonts.set(qn('w:eastAsia'),'宋体');props.append(fonts)
            if size:z=OxmlElement('w:sz');z.set(qn('w:val'),str(round(size*2)));props.append(z)
            r.append(props);tt=OxmlElement('w:t');tt.text=m[1];r.append(tt);h.append(r);p._p.append(h)
        else:
            bold=t.startswith('**') and t.endswith('**');r=p.add_run(t[2:-2] if bold else t);run_font(r,size,bold)
    return p

def columns(section,num):
    cols=section._sectPr.find(qn('w:cols'))
    if cols is None:cols=OxmlElement('w:cols');section._sectPr.append(cols)
    cols.set(qn('w:num'),str(num));cols.set(qn('w:space'),'454')

def is_span(block):
    kind,val=block
    return kind in ('table','image','math') or (kind=='paragraph' and bool(re.match(r'(\*\*(表\d+|Table \d+)|图\d+\s|Fig\. \d+)',val)))

def make_docx(blocks):
    doc=Document();sec=doc.sections[0]
    sec.page_width=Cm(21);sec.page_height=Cm(29.7);sec.left_margin=sec.right_margin=Cm(2);sec.top_margin=Cm(2);sec.bottom_margin=Cm(2)
    sec.header_distance=sec.footer_distance=Cm(.9)
    for name,size,bold in [('Normal',10.5,False),('Title',16,True),('Heading 1',14,True),('Heading 2',10.5,True)]:
        st=doc.styles[name];st.font.name='Times New Roman';st.font.size=Pt(size);st.font.bold=bold;st.element.rPr.rFonts.set(qn('w:eastAsia'),'宋体')
        st.paragraph_format.line_spacing=1.18;st.paragraph_format.space_after=Pt(4)
        if name.startswith('Heading'):st.paragraph_format.keep_with_next=True;st.font.color.rgb=None;st.paragraph_format.space_before=Pt(8)
    header=sec.header.paragraphs[0];inline(header,'视觉单目标跟踪中的模板更新与记忆机制综述',8)
    footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
    fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');footer._p.append(fld)
    current_cols=1;body=False;refs=False;eq=0;data_tables=0
    for idx,(kind,val) in enumerate(blocks):
        if kind=='heading' and val[1].startswith('0 引言'):
            body=True;sec=doc.add_section(WD_SECTION_START.NEW_PAGE);columns(sec,2);current_cols=2
        span=is_span((kind,val))
        if body:
            want=1 if span else 2
            if want!=current_cols:
                sec=doc.add_section(WD_SECTION_START.CONTINUOUS);columns(sec,want);current_cols=want
                empty=doc.paragraphs[-1];empty.paragraph_format.space_after=Pt(0);empty.paragraph_format.space_before=Pt(0);empty.paragraph_format.line_spacing=Pt(1)
        if kind=='heading':
            lv,t=val;refs=t.startswith('参考文献') if lv==2 else refs
            if lv==2 and t.startswith(('附录','作者信息')):refs=False
            if not body and lv==2:p=doc.add_paragraph();inline(p,t,9);p.runs[0].bold=True
            else:p=doc.add_paragraph(style='Title' if lv==1 else 'Heading '+str(lv-1));inline(p,t)
            if lv==1:p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        elif kind=='paragraph':
            p=doc.add_paragraph();size=7.5 if refs else 9 if not body or span else 10.5;inline(p,val,size)
            p.paragraph_format.widow_control=True
            if span:
                p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.keep_with_next=bool(val.startswith(('**表','**Table')))
            elif refs:
                p.paragraph_format.left_indent=Pt(10.5);p.paragraph_format.first_line_indent=Pt(-10.5);p.paragraph_format.space_after=Pt(5)
            elif body:p.paragraph_format.first_line_indent=Pt(21);p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
            elif val.startswith(('作者：','单位：','Template updating','Authors and affiliations')):p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        elif kind=='image':
            p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.add_run().add_picture(str(Path(val[1])),width=Cm(16.4));p.paragraph_format.keep_with_next=True
        elif kind=='math':
            eq+=1;p=doc.add_paragraph();p.paragraph_format.tab_stops.add_tab_stop(Cm(8.1),WD_TAB_ALIGNMENT.CENTER);p.paragraph_format.tab_stops.add_tab_stop(Cm(16.6),WD_TAB_ALIGNMENT.RIGHT)
            p.add_run('\t');p._p.append(el('oMath',math_nodes(eq)));inline(p,'\t('+str(eq)+')',10.5)
        elif kind=='table':
            table=doc.add_table(rows=0,cols=len(val[0]));data_tables+=1;table.autofit=False
            widths=[16.6/len(val[0])]*len(val[0])
            if data_tables==9:widths=[2.8,.9,4.0,8.9]
            for c,w in zip(table.columns,widths):c.width=Cm(w)
            pr=table._tbl.tblPr;borders=OxmlElement('w:tblBorders')
            for edge in ['top','bottom','left','right','insideH','insideV']:
                b=OxmlElement('w:'+edge);b.set(qn('w:val'),'single' if edge in ['top','bottom'] else 'nil');b.set(qn('w:sz'),'8');borders.append(b)
            pr.append(borders)
            for ri,row in enumerate(val):
                cells=table.add_row().cells
                for c,w,content in zip(cells,widths,row):
                    c.width=Cm(w);p=c.paragraphs[0];inline(p,content,7.5);p.paragraph_format.space_after=Pt(3);p.paragraph_format.space_before=Pt(3);p.paragraph_format.line_spacing=1.1
                    if ri==0:
                        for rr in p.runs:rr.bold=True
                        bp=OxmlElement('w:tcBorders');b=OxmlElement('w:bottom');b.set(qn('w:val'),'single');b.set(qn('w:sz'),'6');bp.append(b);c._tc.get_or_add_tcPr().append(bp)
                tr=table.rows[-1]._tr.get_or_add_trPr()
                if ri==0:tr.append(OxmlElement('w:tblHeader'))
                if max(map(len,row))<600:tr.append(OxmlElement('w:cantSplit'))
            doc.add_paragraph().paragraph_format.space_after=Pt(2)
    doc.core_properties.title='视觉单目标跟踪中的模板更新与记忆机制综述';doc.core_properties.author='';doc.core_properties.subject='期刊格式修订稿；作者资料及部分书目信息仍待核定'
    doc.save(ROOT/(NAME+'.docx'))
    return {'editable_equations':eq,'tables':data_tables,'sections':len(doc.sections)}

def ih(t):
    out=[]
    for s in re.split(r'(\*\*.*?\*\*|\[[^\]]+\]\([^)]+\))',t):
        m=re.fullmatch(r'\[([^\]]+)\]\(([^)]+)\)',s)
        if m:out.append('<a href="'+html.escape(m[2],quote=True)+'">'+html.escape(m[1])+'</a>')
        elif s.startswith('**') and s.endswith('**'):out.append('<strong>'+html.escape(s[2:-2])+'</strong>')
        else:out.append(html.escape(s))
    return ''.join(out)

def make_html(blocks):
    parts=['<div class="front">'];body=False;refs=False;eq=0;table_number=0
    for i,(kind,val) in enumerate(blocks):
        if kind=='heading':
            level,title=val
            if title.startswith('0 引言'):parts.append('</div><main>');body=True
            if title.startswith('参考文献'):
                parts.append('</main><main class="references">');refs=True
            if title.startswith('附录'):
                parts.append('</main><main>');refs=False
            if title.startswith('作者信息'):refs=False
            tag='h'+str(level);parts.append(f'<{tag}>{ih(title)}</{tag}>')
        elif kind=='paragraph':
            cap=is_span((kind,val));cls='caption' if cap else 'ref' if refs else 'prose'
            if not body and val.startswith(('作者：','单位：','Template updating','Authors and affiliations')):cls='front-center'
            parts.append(f'<p class="{cls}">{ih(val)}</p>')
        elif kind=='image':parts.append('<div class="wide figure"><img src="'+Path(val[1]).as_uri()+'"></div>')
        elif kind=='math':
            eq+=1;f=plt.figure(figsize=(10,.6));f.text(.5,.5,'$'+val+'$',ha='center',va='center',fontsize=16)
            path=FIG/f'equation_{eq}.png';f.savefig(path,dpi=450,bbox_inches='tight',pad_inches=.1,facecolor='white');plt.close(f)
            parts.append(f'<div class="wide equation"><img src="{path.as_uri()}"><span>({eq})</span></div>')
        elif kind=='table':
            table_number+=1
            cols='<colgroup><col style="width:17%"><col style="width:5%"><col style="width:24%"><col style="width:54%"></colgroup>' if table_number==9 else ''
            parts.append('<table class="wide">'+cols+'<thead><tr>'+''.join('<th>'+ih(x)+'</th>' for x in val[0])+'</tr></thead><tbody>')
            for row in val[1:]:parts.append('<tr>'+''.join('<td>'+ih(x)+'</td>' for x in row)+'</tr>')
            parts.append('</tbody></table>')
    parts.append('</main>')
    css='''
@page {size:A4; margin:18mm 17mm 19mm;}
*{box-sizing:border-box}body{margin:0;color:#111;font-family:"Times New Roman","SimSun",serif;font-size:10.5pt;line-height:1.35}
.front{break-after:page;font-size:9pt}main{column-count:2;column-gap:8mm;column-fill:balance}
h1{font-size:16pt;text-align:center;line-height:1.5;margin:5mm 0}h2{font-size:14pt;margin:5mm 0 2mm;break-after:avoid;line-height:1.3}h3{font-size:10.5pt;margin:4mm 0 1.5mm;break-after:avoid}
p{margin:0 0 2.4mm;text-align:justify;orphans:2;widows:2}.prose{text-indent:2em}.front p{text-indent:0}.front h2{font-size:10.5pt}.front-center{text-align:center}
.wide,.caption{column-span:all}.caption{text-align:center;font-size:9pt;line-height:1.3;text-indent:0;break-after:avoid;margin-top:2mm}
.figure{break-inside:avoid;text-align:center;margin:3mm 0 1mm}.figure img{width:100%;max-height:90mm;object-fit:contain}
.equation{position:relative;text-align:center;margin:3mm 0;break-inside:avoid}.equation img{max-width:92%;max-height:14mm;object-fit:contain}.equation span{position:absolute;right:0;top:50%;transform:translateY(-50%);font-size:10pt}
table{border-collapse:collapse;width:100%;font-size:7.5pt;line-height:1.25;border-top:1pt solid #111;border-bottom:1pt solid #111;margin:2mm 0 4mm;table-layout:fixed}thead{display:table-header-group}th{border-bottom:.6pt solid #111;text-align:left}td,th{padding:1.4mm 1.2mm;vertical-align:top;overflow-wrap:anywhere}tr{break-inside:avoid}
.references{break-before:page}.references p{font-size:7.5pt;line-height:1.3;word-wrap:break-word;overflow-wrap:anywhere;text-indent:-1.3em;padding-left:1.3em;break-inside:avoid}a{color:inherit;text-decoration:none}
'''
    (ROOT/(NAME+'.html')).write_text('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>视觉单目标跟踪中的模板更新与记忆机制综述</title><style>'+css+'</style></head><body>'+''.join(parts)+'</body></html>',encoding='utf-8')

if __name__=='__main__':
    generate_figures();text=(ROOT/(NAME+'.md')).read_text(encoding='utf-8');blocks=parse(text)
    result=make_docx(blocks);make_html(blocks)
    (ROOT/'render_validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
