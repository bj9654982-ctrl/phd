from pathlib import Path
from collections import Counter,defaultdict
import json,re,unicodedata,csv,sys
from openpyxl import Workbook
from openpyxl.styles import Font,PatternFill,Alignment
from openpyxl.worksheet.table import Table,TableStyleInfo
from selection_plan import GROUPS

sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parent
parent=ROOT.parent
oldtext=(parent/'03_综述完整初稿_v1_20261025.md').read_text(encoding='utf-8')
discovered=json.loads((ROOT/'candidates.json').read_text(encoding='utf-8'))
selected=json.loads((ROOT/'selected_enriched.json').read_text(encoding='utf-8'))
extras=json.loads((ROOT/'extra_records.json').read_text(encoding='utf-8'))

def norm(s):
    return ''.join(c for c in unicodedata.normalize('NFKD',s).lower() if c.isalnum())

oldcat={1:'更新策略与在线适应',2:'相关综述与研究定位',3:'相关综述与研究定位',4:'相关综述与研究定位',5:'相关综述与研究定位',6:'相关综述与研究定位',
7:'相关滤波与样本正则化',8:'相关滤波与样本正则化',9:'孪生匹配与目标条件化',10:'相关滤波与样本正则化',11:'长时跟踪与记忆恢复',
12:'注意力与时序状态建模',13:'更新策略与在线适应',14:'更新策略与在线适应',15:'更新策略与在线适应',16:'注意力与时序状态建模',17:'注意力与时序状态建模',
18:'注意力与时序状态建模',19:'注意力与时序状态建模',20:'注意力与时序状态建模',21:'注意力与时序状态建模',22:'注意力与时序状态建模',23:'注意力与时序状态建模',
24:'注意力与时序状态建模',25:'注意力与时序状态建模',26:'注意力与时序状态建模',27:'注意力与时序状态建模',28:'长时跟踪与记忆恢复',29:'长时跟踪与记忆恢复',
30:'数据集与评价设计',31:'数据集与评价设计',32:'数据集与评价设计',33:'数据集与评价设计',34:'数据集与评价设计',35:'数据集与评价设计'}
bytitle={norm(r['title']):r for r in discovered}
records=[]
for line in oldtext.splitlines():
    m=re.match(r'^\[(\d+)\]\s+(.*?)\.\s+(.*?)\[([JC])\](.*)$',line)
    if not m:continue
    num=int(m[1]);authors=[x.strip() for x in m[2].split(',')];title=m[3];tail=m[5]
    links=re.findall(r'\[[^\]]+\]\(([^)]+)\)',line)
    year=int(re.search(r'\b(?:19|20)\d{2}\b',tail)[0])
    vm=re.search(r'^\.\s*(.*?),\s*\d{4}',tail) if m[4]=='J' else re.search(r'^//(.*?)\.',tail)
    venue=vm[1] if vm else ''
    doi_m=re.search(r'DOI:\s*(\S+)',line)
    vi=re.search(r'\b'+str(year)+r',\s*(\d+)(?:\(([^)]+)\))?:\s*([^\.]+)',tail)
    row={'key':f'R{num}','number':num,'title':title,'authors':authors,'year':year,'venue':venue,
         'type':'journal' if m[4]=='J' else 'conference','url':links[0] if links else '',
         'doi':doi_m[1].rstrip('.') if doi_m else '', 'volume':vi[1] if vi else '',
         'issue':(vi[2] or '') if vi else '', 'pages':vi[3] if vi else '',
         'category':oldcat[num],'verification':'v1 primary-source record retained; matched official proceedings metadata refreshed where available',
         'reference_original':line,'all_source_urls':links,'access_date':'2026-10-25'}
    if norm(title) in bytitle:
        match=bytitle[norm(title)]
        for field in ['authors','title','pages','url','pdf','bibtex_original','source_index']:
            if match.get(field):row[field]=match[field]
        if match.get('doi'):row['doi']=match['doi']
    records.append(row)
assert len(records)==35
records.extend(extras)
byid={r['candidate_id']:r for r in selected}
for cat,ids in GROUPS.items():
    for cid in sorted(ids,key=lambda x:(byid[x]['year'],byid[x]['title'].lower())):
        r=dict(byid[cid]);r['key']='P'+str(cid);r['type']='conference';records.append(r)
for i,r in enumerate(records,1):r['number']=i
assert len(records)==300,len(records)
duplicates=[t for t,n in Counter(norm(r['title']) for r in records).items() if n>1]
assert not duplicates,duplicates
dupedois=[t for t,n in Counter(r.get('doi','').lower() for r in records if r.get('doi')).items() if n>1]
assert not dupedois,dupedois

P1={f'R{i}' for i in [1,2,7,8,9,10,11,12,13,14,15,17,18,19,20,21,22,23,24,25,26,27,28,29,34,35]}
P1|={'C3','C5','C6','C8','C10','C11','S1','S2'}
P1|={f'P{i}' for i in [147,156,204,283,389,407,630,741,782,833,826,944,634,781,916,873]}
assert len(P1)==50
sections={
'相关综述与研究定位':'第0章；第4.6节','传统外观建模与局部可靠性':'第1.2、2.6节',
'相关滤波与样本正则化':'第1.2、2.6节','孪生匹配与目标条件化':'第2.1—2.5、2.7节',
'更新策略与在线适应':'第2.2—2.4、2.6—2.8节','注意力与时序状态建模':'第3.1—3.7节',
'自回归与生成式跟踪':'第3.7节','自监督训练与表征迁移':'第2.8、5.6节',
'长时跟踪与记忆恢复':'第2.4、3.6节','高效跟踪与资源预算':'第3.8、5.3节',
'无人机与退化观测':'第3.8、4.1节','语言条件与语义更新':'第3.9节',
'多模态历史与统一跟踪':'第3.9节','对抗扰动与可靠性诊断':'第2.8、4.5节',
'视频分割中的可迁移记忆机制':'第3.10节','数据集与评价设计':'第4章'}
questions={
'相关综述与研究定位':'它以架构、任务还是更新机制组织文献？本文主题是否重复？',
'传统外观建模与局部可靠性':'哪些局部外观被保留？遮挡后哪些样本或部件需要减权？',
'相关滤波与样本正则化':'历史样本权重、空间约束和在线模型更新怎样共同抑制漂移？',
'孪生匹配与目标条件化':'提升来自匹配表示、定位头还是历史更新？是否保存跨帧状态？',
'更新策略与在线适应':'实际更新的是图像模板、特征、分类器还是模型参数？何时更新？',
'注意力与时序状态建模':'当前帧怎样查询历史？如何控制写入与读出的错误信息？',
'自回归与生成式跟踪':'递归输入包含什么状态？训练是否覆盖自身预测误差？',
'自监督训练与表征迁移':'伪标签或跨帧对应中的错误怎样影响训练？与测试时更新有何区别？',
'长时跟踪与记忆恢复':'目标丢失后如何识别错误、检索可信历史并恢复定位？',
'高效跟踪与资源预算':'压缩的是模型、模板token还是历史容量？速度是否包含更新与读取？',
'无人机与退化观测':'模糊、视角和延迟怎样影响历史可靠性与状态预测？',
'语言条件与语义更新':'语言初始化和边界框初始化是否相同？语义更新如何维持实例身份？',
'多模态历史与统一跟踪':'模态缺失时哪一份历史可信？融合、参数适应与记忆更新如何区分？',
'对抗扰动与可靠性诊断':'扰动改变当前定位还是污染后续历史？评价是否区分二者？',
'视频分割中的可迁移记忆机制':'什么记忆组织可迁移到边界框跟踪？掩码监督、交互和未来信息是否改变协议？',
'数据集与评价设计':'初始化、目标消失、重置、属性、硬件和指标聚合怎样影响比较？'}

for r in records:
    cat=r['category'];r['section']=sections[cat];r['reading_question']=questions[cat]
    r['priority']='P1 核心精读' if r['key'] in P1 else 'P2 对照阅读' if r['year']>=2017 and cat not in ['语言条件与语义更新','多模态历史与统一跟踪','对抗扰动与可靠性诊断'] else 'P3 背景与边界'
    r['reading_status']='未标记为全文精读；元数据核对与阅读完成度分别记录'
    r['scope']='机制迁移参照：视频分割' if cat=='视频分割中的可迁移记忆机制' or r['key'] in ['P146','P305','P703','P708'] else '协议扩展：语言或多模态' if cat in ['语言条件与语义更新','多模态历史与统一跟踪'] else 'SOT方法、基础或评价'
    r['pages']=r.get('pages','').replace('--','–').replace('-','–')

def reference(r):
    if r['key'].startswith('R'):return r['reference_original']
    authors=r['authors'];chinese=bool(re.search(r'[\u4e00-\u9fff]',r['title']))
    auth=', '.join(authors[:3])+(', 等' if chinese else ', et al.') if len(authors)>3 else ', '.join(authors)
    j=r['type']=='journal'
    s=f"[{r['number']}] {auth}. {r['title']}"+('[J]. ' if j else '[C]//')+r['venue']+f". {r['year']}"
    if r.get('volume'):s+=', '+str(r['volume'])
    if r.get('issue'):s+='('+str(r['issue'])+')'
    if r.get('pages'):s+=': '+r['pages']
    s+='.'
    if r.get('doi'):s+=' DOI: '+r['doi']+'.'
    s+=' [原始来源]('+r['url']+').'
    return s

for r in records:r['formatted_reference']=reference(r)
(ROOT/'library_300.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
(ROOT/'references_300.md').write_text('\n\n'.join(r['formatted_reference'] for r in records)+'\n',encoding='utf-8')

columns={'编号':'number','检索键':'key','题名':'title','作者':'authors','年份':'year','出版物':'venue','卷':'volume','期':'issue','页码':'pages','DOI':'doi','来源链接':'url','主题分类':'category','适用范围':'scope','正文位置':'section','阅读优先级':'priority','精读问题':'reading_question','核验层级':'verification','精读状态':'reading_status'}
table=[]
for r in records:
    table.append({cn:('; '.join(r.get(k,[])) if k=='authors' else r.get(k,'')) for cn,k in columns.items()})
with (ROOT/'02_300篇文献分类索引.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(table)

wb=Workbook();wb.remove(wb.active)
for name,items in [('全部300篇',table),('优先精读50篇',[x for x in table if x['阅读优先级'].startswith('P1')])]:
    ws=wb.create_sheet(name);ws.append(list(columns))
    for item in items:ws.append(list(item.values()))
    ws.freeze_panes='D2';ws.auto_filter.ref=ws.dimensions
    for cell in ws[1]:cell.font=Font(color='FFFFFF',bold=True);cell.fill=PatternFill('solid',fgColor='19354A')
    widths=[8,10,66,35,8,34,8,8,15,34,60,28,31,28,22,65,48,55]
    for i,w in enumerate(widths,1):ws.column_dimensions[ws.cell(1,i).column_letter].width=w
    for row in ws.iter_rows(min_row=2):
        for cell in row:cell.alignment=Alignment(vertical='top',wrap_text=True)
        ws.row_dimensions[row[0].row].height=45
        link=row[10];link.hyperlink=link.value;link.style='Hyperlink'
summary=wb.create_sheet('分类与核验说明');summary.append(['主题','篇数'])
for cat,n in Counter(r['category'] for r in records).items():summary.append([cat,n])
summary.append(['合计',len(records)]);summary.append(['说明','数量为本稿选入文献，不能当作全领域年度发文统计。'])
summary.append(['核验与精读','249篇新增CVF/ECVA文献均保存官方元数据与摘要；这不等于249篇已经逐页精读。'])
summary.append(['版本规则','同一论文的预印本和正式版本只计1条；正式版本优先。原稿35条编号保留。'])
summary.column_dimensions['A'].width=38;summary.column_dimensions['B'].width=105
wb.save(ROOT/'02_300篇文献分类索引.xlsx')

def tex_escape(s):return str(s).replace('&',r'\&').replace('%',r'\%').replace('_',r'\_')
bibs=[];ris=[]
for r in records:
    kind='article' if r['type']=='journal' else 'inproceedings'
    authors=['others' if x.lower() in ['et al.','et al','等'] else x for x in r['authors']]
    fields={'title':r['title'],'author':' and '.join(authors),'year':r['year'],('journal' if kind=='article' else 'booktitle'):r['venue'],'volume':r.get('volume',''),'number':r.get('issue',''),'pages':r.get('pages','').replace('–','--'),'doi':r.get('doi',''),'url':r['url']}
    bibs.append('@'+kind+'{SOT'+str(r['number']).zfill(3)+',\n'+',\n'.join('  '+k+' = {'+tex_escape(v)+'}' for k,v in fields.items() if v!='')+'\n}')
    lines=['TY  - '+('JOUR' if kind=='article' else 'CONF'),f"ID  - SOT{r['number']:03d}",'TI  - '+r['title']]
    lines+=['AU  - '+a for a in authors if a!='others']
    lines+=['PY  - '+str(r['year']),'T2  - '+r['venue'],'UR  - '+r['url']]
    if r.get('doi'):lines.append('DO  - '+r['doi'])
    if r.get('volume'):lines.append('VL  - '+str(r['volume']))
    if r.get('issue'):lines.append('IS  - '+str(r['issue']))
    if r.get('pages'):
        pp=r['pages'].split('–');lines.append('SP  - '+pp[0])
        if len(pp)>1:lines.append('EP  - '+pp[-1])
    lines+=['KW  - '+r['category'],'N1  - '+r['verification']+'; '+r['reading_status'],'ER  - ']
    ris.append('\n'.join(lines))
(ROOT/'03_300篇参考文献.bib').write_text('\n\n'.join(bibs)+'\n',encoding='utf-8')
(ROOT/'03_300篇参考文献.ris').write_text('\n\n'.join(ris)+'\n',encoding='utf-8')
stats={'total':len(records),'unique_titles':len(set(norm(r['title']) for r in records)),'unique_nonempty_dois':len(set(r.get('doi','').lower() for r in records if r.get('doi'))),'year_range':[min(r['year'] for r in records),max(r['year'] for r in records)],'priority':dict(Counter(r['priority'] for r in records)),'categories':dict(Counter(r['category'] for r in records)),'new_official_abstract_records':len(selected)}
(ROOT/'library_validation.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(stats,ensure_ascii=False,indent=2))
