from pathlib import Path
import re, html, json, csv, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager
import fitz
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.opc.constants import RELATIONSHIP_TYPE as RT

sys.stdout.reconfigure(encoding="utf-8")
ROOT=Path(__file__).resolve().parent
SRC=ROOT/"01_综述完整初稿_v2_300篇文献_20261025.md"
FIG=ROOT/"figures"
FIG.mkdir(exist_ok=True)
FONT=font_manager.FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
plt.rcParams.update({"font.family":FONT.get_name(),"axes.unicode_minus":False,"svg.fonttype":"none",
                     "font.size":11,"axes.spines.top":False,"axes.spines.right":False})
NAVY="#19354A"; BLUE="#2D728F"; TEAL="#337C6A"; ORANGE="#BB6A32"; GRAY="#52616C"
def canvas(w=12,h=4):
    f,a=plt.subplots(figsize=(w,h)); a.set_xlim(0,12); a.set_ylim(0,h); a.axis("off"); return f,a
def box(a,x,y,w,h,t,color=BLUE,fs=12):
    a.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.06,rounding_size=0.09",
                             linewidth=1.2,edgecolor=color,facecolor=color+"10"))
    a.text(x+w/2,y+h/2,t,ha="center",va="center",color=NAVY,fontproperties=FONT,fontsize=fs)
def arrow(a,start,end,style="-",color=GRAY,rad=0):
    a.add_patch(FancyArrowPatch(start,end,arrowstyle="-|>",mutation_scale=13,linewidth=1.25,
                               linestyle=style,color=color,connectionstyle=f"arc3,rad={rad}"))
def save(f,name):
    f.savefig(FIG/(name+".png"),dpi=220,bbox_inches="tight",facecolor="white")
    f.savefig(FIG/(name+".svg"),bbox_inches="tight",facecolor="white")
    plt.close(f)

f,a=canvas(12,4.5)
box(a,.25,2.65,1.85,.85,"历史状态\nM(t−1)",TEAL)
box(a,2.65,2.65,1.65,.85,"读取 R",BLUE)
box(a,4.85,2.65,1.65,.85,"目标定位 L",BLUE)
box(a,7.1,2.65,1.85,.85,"候选信息 c(t)",ORANGE)
box(a,9.65,2.65,1.85,.85,"状态更新 U",TEAL)
for st,en in [((2.1,3.08),(2.6,3.08)),((4.3,3.08),(4.8,3.08)),((6.5,3.08),(7.05,3.08)),((8.95,3.08),(9.6,3.08))]:arrow(a,st,en)
box(a,2.65,.95,3.85,.85,"当前图像 / 搜索表征",BLUE)
arrow(a,(3.4,1.85),(3.4,2.6));arrow(a,(5.7,1.85),(5.7,2.6))
box(a,7.1,.95,1.85,.85,"可靠性 q(t)",ORANGE)
arrow(a,(6.5,1.38),(7.05,1.38));arrow(a,(8.95,1.38),(10.55,2.6),rad=.12)
arrow(a,(6.45,2.68),(7.55,1.84),color=GRAY)
a.plot([10.55,10.55,1.18,1.18],[3.55,4.1,4.1,3.55],ls="--",color=TEAL,lw=1.2)
arrow(a,(1.18,3.83),(1.18,3.54),style="--",color=TEAL)
a.text(6,4.22,"更新后的 M(t) 在下一时刻成为可用历史",ha="center",fontsize=11,color=TEAL,fontproperties=FONT)
a.text(6,.35,"因果约束：当前预测只能使用当前图像与过去状态；真值用于离线评价",ha="center",color=GRAY,fontsize=11,fontproperties=FONT)
save(f,"fig1_loop")

f,a=canvas(12,5.1)
titles=["模板表示","显式空间记忆","压缩历史状态"]
descs=["保存初始 / 动态外观参照\n以替换、融合或多模板交互适应变化",
       "保存多个历史特征与位置线索\n根据当前查询选择、读取与融合",
       "将过去上下文递推到 token / 隐状态\n通过状态转换影响下一帧推理"]
ex=["UpdateNet · STARK · MixFormer","MemTrack · STMTrack · HIPTrack","ROMTrack · ODTrack · MCITrack"]
for i in range(3):
    x=.25+4*i
    box(a,x,.8,3.5,3.85,"", [BLUE,TEAL,ORANGE][i])
    a.text(x+1.75,4.15,titles[i],ha="center",fontsize=15,color=NAVY,fontproperties=FONT,weight="bold")
    if i==0:
        for xx,label in [(x+.4,"初始"),(x+1.9,"动态")]:box(a,xx,2.9,1.15,.65,label,BLUE,11)
        arrow(a,(x+1.55,3.23),(x+1.9,3.23))
    elif i==1:
        for j in range(3):box(a,x+.28+j*1.02,2.85,.82,.85,f"历史\n{j+1}",TEAL,10)
    else:
        for j in range(3):
            box(a,x+.25+j*1.05,2.95,.8,.65,["h(t−2)","h(t−1)","h(t)"][j],ORANGE,9)
            if j<2:arrow(a,(x+1.05+j*1.05,3.27),(x+1.29+j*1.05,3.27))
    a.text(x+1.75,2.18,descs[i],ha="center",va="center",fontsize=10.5,color=NAVY,fontproperties=FONT,linespacing=1.6)
    a.text(x+1.75,1.22,ex[i],ha="center",fontsize=9.3,color=GRAY)
a.text(6,.3,"这些表示可以组合；分类维度不等于模型性能等级",ha="center",fontsize=11,color=GRAY,fontproperties=FONT)
save(f,"fig2_memory")

f,a=canvas(12,5.2)
rows=[
("更新可靠性",BLUE,[("2004","模板更新问题"),("2019","UpdateNet"),("2021—2022","STARK / MixFormer"),("2025","UncTrack")]),
("显式历史利用",TEAL,[("2015","MUSTer"),("2018","MemTrack"),("2021—2024","STMTrack / HIPTrack"),("2026","MUTrack")]),
("跨帧状态传递",ORANGE,[("2021","时序上下文"),("2023","ROMTrack"),("2024","ODTrack"),("2025","MCITrack")])]
for i,(label,col,items) in enumerate(rows):
    y=3.75-1.35*i
    a.text(.2,y+.38,label,color=col,fontproperties=FONT,fontsize=12,va="center")
    for j,(year,name) in enumerate(items):
        x=2.05+j*2.45
        box(a,x,y,2.1,.9,year+"\n"+name,col,10.5)
        if j<3:arrow(a,(x+2.1,y+.45),(x+2.4,y+.45),color=col)
a.text(6,.3,"代表工作按所讨论问题并列；跨行机制可共存，非完整时间谱系",ha="center",fontsize=10.5,color=GRAY,fontproperties=FONT)
save(f,"fig3_routes")

f,axs=plt.subplots(1,2,figsize=(11.6,4.2))
vals=[79.1,79.3,80.2,80.3,80.2,79.8]
axs[0].plot(range(6),vals,"o-",color=BLUE,lw=1.8,ms=6)
axs[0].set_xticks(range(6),["1","2","4","6","8","ALL"])
axs[0].set_ylim(78.8,80.65)
axs[0].set_xlabel("推理参考帧数量",fontproperties=FONT);axs[0].set_ylabel("TrackingNet AUC / %",fontproperties=FONT)
axs[0].set_title("STMTrack：参考帧数量",fontproperties=FONT,color=NAVY)
for i,v in enumerate(vals):axs[0].annotate(str(v),(i,v),xytext=(0,8),textcoords="offset points",ha="center",fontsize=10)
v2=[72.7,72.7,72.7,72.6]
axs[1].plot([5,10,20,30],v2,"s-",color=TEAL,lw=1.8,ms=6)
axs[1].set_xticks([5,10,20,30]);axs[1].set_ylim(72.4,72.95)
axs[1].set_xlabel("更新间隔 / 帧",fontproperties=FONT);axs[1].set_ylabel("LaSOT AUC / %",fontproperties=FONT)
axs[1].set_title("HIPTrack：更新间隔",fontproperties=FONT,color=NAVY)
for x,v in zip([5,10,20,30],v2):axs[1].annotate(str(v),(x,v),xytext=(0,8),textcoords="offset points",ha="center",fontsize=10)
for ax in axs:ax.grid(axis="y",alpha=.22);ax.tick_params(labelsize=10)
f.text(.5,.025,"原论文报告值重新绘制：STMTrack 表4；HIPTrack 补充材料表3。非本文复现实验。",ha="center",fontproperties=FONT,fontsize=9.4,color=GRAY)
f.tight_layout(rect=(0,.08,1,1));save(f,"fig4_evidence")

text=SRC.read_text(encoding="utf-8")
# The parser preserves paragraphs, editable tables, headings and hyperlink text.
def blocks_of(t):
    lines=t.splitlines(); i=0
    while i<len(lines):
        s=lines[i].strip()
        if not s: i+=1;continue
        if s==r"\[":
            k=i+1;eq=[]
            while k<len(lines) and lines[k].strip()!=r"\]":eq.append(lines[k].strip());k+=1
            yield ("math"," ".join(eq));i=k+1;continue
        if s.startswith("|"):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith("|"):
                cells=[c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"[:\- ]+",c) for c in cells):rows.append(cells)
                i+=1
            yield("table",rows);continue
        m=re.match(r"^(#{1,3}) (.*)",s)
        if m:yield("heading",(len(m[1]),m[2]));i+=1;continue
        m=re.match(r"^!\[(.*?)\]\((.*?)\)$",s)
        if m:yield("image",(m[1],m[2]));i+=1;continue
        if s.startswith(">"):
            q=[]
            while i<len(lines) and lines[i].lstrip().startswith(">"):
                q.append(lines[i].lstrip()[1:].strip());i+=1
            yield("quote"," ".join(q));continue
        parts=[s];i+=1
        while i<len(lines) and lines[i].strip() and not re.match(r"^(#{1,3} |[>|]|!\\[|\\\[)",lines[i].strip()):
            parts.append(lines[i].strip());i+=1
        yield("paragraph"," ".join(parts))
blocks=list(blocks_of(text))
eqpaths={}
for idx,(typ,val) in enumerate(blocks):
    if typ=="math":
        f=plt.figure(figsize=(10,.55));f.text(.5,.5,"$"+val+"$",ha="center",va="center",fontsize=16)
        p=FIG/f"eq_{idx}.png";f.savefig(p,dpi=230,bbox_inches="tight",pad_inches=.08,transparent=False,facecolor="white");plt.close(f);eqpaths[idx]=p

doc=Document()
sec=doc.sections[0]
sec.page_width=Cm(21);sec.page_height=Cm(29.7)
sec.top_margin=Cm(2.1);sec.bottom_margin=Cm(2);sec.left_margin=Cm(2);sec.right_margin=Cm(2)
sec.header_distance=Cm(.9);sec.footer_distance=Cm(.9)
styles=doc.styles
for name in ["Normal","Body Text"]:
    st=styles[name];st.font.name="Times New Roman";st.font.size=Pt(11)
    st.element.rPr.rFonts.set(qn("w:eastAsia"),"宋体")
    st.paragraph_format.line_spacing=1.35;st.paragraph_format.space_after=Pt(6)
for name,size in [("Title",18),("Heading 1",15),("Heading 2",12.5)]:
    st=styles[name];st.font.name="Times New Roman";st.font.size=Pt(size)
    st.font.color.rgb=RGBColor.from_string("19354A");st.element.rPr.rFonts.set(qn("w:eastAsia"),"微软雅黑")
    st.paragraph_format.space_before=Pt(13);st.paragraph_format.space_after=Pt(7)
    st.paragraph_format.keep_with_next=True
h=sec.header.paragraphs[0];h.text="视觉单目标跟踪：模板更新与记忆机制 · 阅读初稿"
h.runs[0].font.size=Pt(8);h.runs[0].font.name="Times New Roman"
h.runs[0]._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"),"宋体")
foot=sec.footer.paragraphs[0];foot.alignment=WD_ALIGN_PARAGRAPH.CENTER
r=foot.add_run("— ");r.font.size=Pt(9)
field=OxmlElement("w:fldSimple");field.set(qn("w:instr"),"PAGE");foot._p.append(field);foot.add_run(" —")
def add_link(par,label,url):
    rel=par.part.relate_to(url,RT.HYPERLINK,is_external=True)
    hyp=OxmlElement("w:hyperlink");hyp.set(qn("r:id"),rel)
    rr=OxmlElement("w:r");prop=OxmlElement("w:rPr")
    color=OxmlElement("w:color");color.set(qn("w:val"),"2D728F");prop.append(color)
    rr.append(prop);tt=OxmlElement("w:t");tt.text=label;rr.append(tt);hyp.append(rr);par._p.append(hyp)
def inline_doc(par,s):
    pat=r"(\*\*.*?\*\*|\[[^\]]+\]\([^)]+\))"
    for item in re.split(pat,s):
        if item.startswith("**") and item.endswith("**"):par.add_run(item[2:-2]).bold=True
        else:
            m=re.fullmatch(r"\[([^\]]+)\]\(([^)]+)\)",item)
            if m:add_link(par,m[1],m[2])
            else:par.add_run(item)
    for rr in par.runs:
        if rr._element.get_or_add_rPr().rFonts is None:
            fonts=OxmlElement("w:rFonts");rr._element.rPr.append(fonts)
        rr._element.rPr.rFonts.set(qn("w:eastAsia"),"宋体")

for idx,(typ,val) in enumerate(blocks):
    if typ=="heading":
        level,title=val;p=doc.add_paragraph(style="Title" if level==1 else "Heading "+str(level-1));inline_doc(p,title)
        if level==1:p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    elif typ in ("paragraph","quote"):
        p=doc.add_paragraph();inline_doc(p,val)
        if typ=="quote":
            for rr in p.runs:rr.font.size=Pt(9);rr.font.color.rgb=RGBColor.from_string("52616C")
        elif val.startswith("图") or val.startswith("注：") or val.startswith("来源：") or val.startswith("数据来源："):
            for rr in p.runs:rr.font.size=Pt(9)
        elif val.startswith("**表"):
            p.paragraph_format.keep_with_next=True
        elif not val.startswith("["):p.paragraph_format.first_line_indent=Pt(22)
    elif typ in ("image","math"):
        p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        imgpath=Path(val[1]) if typ=="image" else eqpaths[idx]
        pix=fitz.Pixmap(str(imgpath));width=min(6.5, pix.width/220 if typ=="image" else pix.width/230)
        p.add_run().add_picture(str(imgpath),width=Inches(width))
    elif typ=="table":
        table=doc.add_table(rows=0,cols=len(val[0]));table.style="Table Grid"
        for ri,row in enumerate(val):
            cells=table.add_row().cells
            for ci,content in enumerate(row):
                p=cells[ci].paragraphs[0];inline_doc(p,content)
                p.paragraph_format.space_after=Pt(3);p.paragraph_format.space_before=Pt(3);p.paragraph_format.line_spacing=1.12
                for rr in p.runs:rr.font.size=Pt(8.5);rr.bold=(ri==0)
                if ri==0:
                    shade=OxmlElement("w:shd");shade.set(qn("w:fill"),"EAF1F5");cells[ci]._tc.get_or_add_tcPr().append(shade)
            trpr=table.rows[-1]._tr.get_or_add_trPr()
            cant=OxmlElement("w:cantSplit");trpr.append(cant)
            if ri==0:
                repeat=OxmlElement("w:tblHeader");trpr.append(repeat)
        doc.add_paragraph()
doc.core_properties.title="视觉单目标跟踪中的模板更新与记忆机制综述"
doc.core_properties.subject="文献分析阅读初稿，原文报告值整理"
doc.core_properties.author=""
doc.save(ROOT/"01_综述完整初稿_v2_300篇文献_20261025.docx")

def inline_html(s):
    # Escape prose while retaining actual link destinations and strong text.
    out=[]
    for item in re.split(r"(\*\*.*?\*\*|\[[^\]]+\]\([^)]+\))",s):
        if item.startswith("**") and item.endswith("**"):out.append("<strong>"+html.escape(item[2:-2])+"</strong>")
        else:
            m=re.fullmatch(r"\[([^\]]+)\]\(([^)]+)\)",item)
            if m:out.append('<a href="'+html.escape(m[2],quote=True)+'">'+html.escape(m[1])+"</a>")
            else:out.append(html.escape(item))
    return "".join(out)
parts=[]
for idx,(typ,val) in enumerate(blocks):
    if typ=="heading":
        lv,title=val;parts.append(f'<h{lv} id="h{idx}">'+inline_html(title)+f'</h{lv}>')
    elif typ in ("paragraph","quote"):
        cls="note" if typ=="quote" else "caption" if val.startswith(("图","注：","来源：","数据来源：","**表")) else "ref" if re.match(r"^\[\d+\]",val) else ""
        if val.startswith("**表8 "):
            cls += " newpage"
        parts.append(f'<p class="{cls}">'+inline_html(val)+"</p>")
    elif typ in ("image","math"):
        p=Path(val[1]) if typ=="image" else eqpaths[idx]
        rel=p.relative_to(ROOT).as_posix()
        pix=fitz.Pixmap(str(p))
        width=min(480, pix.width/(220 if typ=="image" else 230)*72)
        parts.append(f'<div class="figure"><img src="{rel}" style="width:{width:.1f}pt"/></div>')
    elif typ=="table":
        parts.append('<div class="tablewrap"><table>')
        for ri,row in enumerate(val):
            parts.append("<tr>")
            for c in row:
                tag="th" if ri==0 else "td";parts.append(f"<{tag}>"+inline_html(c)+f"</{tag}>")
            parts.append("</tr>")
        parts.append("</table></div>")
css="""
@font-face {font-family: Song; src: url('simsun.ttc');}
@font-face {font-family: YaHei; src: url('msyh.ttc');}
body {font-family: Song; font-size:10.5pt; line-height:1.55; color:#172C3C;}
h1,h2,h3 {font-family:YaHei; color:#19354A; page-break-after:avoid;}
h1 {font-size:18pt; text-align:center; margin:8pt 0 14pt;}
h2 {font-size:14pt; margin:16pt 0 8pt;}
h3 {font-size:11.5pt; margin:12pt 0 6pt;}
p {margin:0 0 7pt; text-indent:21pt; text-align:justify;}
p.note {background:#F0F4F6; font-size:9pt; text-indent:0;}
p.caption {font-size:9pt; text-indent:0; color:#40545D;}
p.newpage {page-break-before:always;}
p.ref {font-size:9pt; text-indent:0; margin-bottom:7pt;}
a {color:#2D728F;}
table {border-collapse:collapse; width:100%; font-size:8pt; line-height:1.35; margin:8pt 0 10pt;}
.tablewrap {page-break-inside:avoid;}
th,td {border:0.5pt solid #A6B5BD; padding:4pt; vertical-align:top;}
th {font-weight:bold;}
tr {page-break-inside:avoid;}
.figure {text-align:center; margin:8pt 0; page-break-inside:avoid;}
"""
htmltext="<html><head></head><body>"+"\n".join(parts)+"</body></html>"
(ROOT/"01_综述完整初稿_v2_300篇文献_20261025.html").write_text("<html><head><meta charset='utf-8'><style>"+css+"</style></head><body>"+"\n".join(parts)+"</body></html>",encoding="utf-8")
archive=fitz.Archive(str(ROOT));archive.add("C:/Windows/Fonts")
story=fitz.Story(html=htmltext,user_css=css,archive=archive)
pdf_tmp=ROOT/"_reading_layout.pdf"
writer=fitz.DocumentWriter(str(pdf_tmp))
page_rect=fitz.Rect(0,0,595.28,841.89)
def rectfn(n,filled):
    if n>120:raise RuntimeError("Unexpected pagination count")
    return page_rect,fitz.Rect(48,50,547.28,791.89),None
positions=[]
story.write(writer,rectfn,positionfn=lambda p:positions.append((p.page_num,p.heading,p.text,p.open_close)) if p.heading else None)
writer.close()
del writer
pdf=fitz.open(pdf_tmp)
for i,page in enumerate(pdf):
    page.insert_text((48,28),"Template Updating and Memory in Visual Tracking | Reading Draft",fontsize=8,color=(.3,.38,.43))
    page.insert_text((285,819),f"{i+1} / {len(pdf)}",fontsize=8,color=(.3,.38,.43))
toc=[]
seen=set()
for page,heading,txt,oc in positions:
    if txt and (txt,page) not in seen and heading<=3:
        seen.add((txt,page));toc.append([max(1,heading-1),txt,page])
if toc:
    # Ensure the outline starts at level one and never skips a level.
    for i,row in enumerate(toc):
        if i==0:row[0]=1
        else:row[0]=min(row[0],toc[i-1][0]+1)
    pdf.set_toc(toc)
pdf.set_metadata({"title":"视觉单目标跟踪中的模板更新与记忆机制综述","subject":"文献分析阅读初稿；无新增复现实验","author":""})
pdf.subset_fonts()
out=ROOT/"01_综述完整初稿_v2_300篇文献_20261025.pdf"
pdf.save(out,garbage=4,deflate=True);pdf.close()
try:
    pdf_tmp.unlink()
except PermissionError:
    pass
# Machine-readable source index and numerical evidence for later checking.
refs=[]
for line in text.splitlines():
    m=re.match(r"^\[(\d+)\]\s+(.*)",line)
    if m:
        links=re.findall(r"\[([^\]]+)\]\(([^)]+)\)",m[2])
        refs.append({"编号":m[1],"书目信息":re.sub(r"\[([^\]]+)\]\(([^)]+)\)",r"\1",m[2]),"链接":" | ".join(u for _,u in links)})
with (ROOT/"05_参考文献索引.csv").open("w",encoding="utf-8-sig",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=["编号","书目信息","链接"]);w.writeheader();w.writerows(refs)
evidence=[
{"source":"STMTrack CVPR2021 Table4","variable":"reference_frames","settings":["1","2","4","6","8","ALL"],"dataset":"TrackingNet","AUC_percent":[79.1,79.3,80.2,80.3,80.2,79.8],"FPS":[43,26.6,29.3,28.6,22.7,6.5],"reproduced":False},
{"source":"HIPTrack CVPR2024 supplementary Table3","variable":"update_interval","settings":[5,10,20,30],"dataset":"LaSOT","AUC_percent":[72.7,72.7,72.7,72.6],"PNorm_percent":[82.9,82.9,82.9,82.5],"P_percent":[79.5,79.5,79.5,79],"reproduced":False}]
(ROOT/"06_文献报告数据.json").write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding="utf-8")
with fitz.open(out) as check:
    extracted="\n".join(p.get_text() for p in check)
    counts=[len(p.get_text().strip()) for p in check]
    check[0].get_pixmap(matrix=fitz.Matrix(1.3,1.3)).save(ROOT/"preview_page1.png")
    for p in check:
        if "表7" in p.get_text() and "79.1" in p.get_text():
            p.get_pixmap(matrix=fitz.Matrix(1.3,1.3)).save(ROOT/"preview_evidence.png");break
    stats={"body_hanzi":len(re.findall(r"[\u4e00-\u9fff]",text.split("## 参考文献")[0])),
           "references":len(refs),"main_figures":5,"tables":len([b for b in blocks if b[0]=="table"]),
           "pdf_pages":len(check),"pdf_text_chars":len(extracted),"min_page_text_chars":min(counts),
           "pdf_references_last_present":"[300]" in extracted,"pdf_end_present":"原论文报告值未由本稿重新验证" in extracted,
           "docx_bytes":(ROOT/"01_综述完整初稿_v2_300篇文献_20261025.docx").stat().st_size,
           "pdf_bytes":out.stat().st_size}
(ROOT/"validation.json").write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(stats,ensure_ascii=False,indent=2))

