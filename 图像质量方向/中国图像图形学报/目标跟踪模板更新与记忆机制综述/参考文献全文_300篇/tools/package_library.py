from pathlib import Path
import json,csv,sys,html,hashlib,shutil,importlib.util,collections
from urllib.parse import quote
from openpyxl import Workbook
from openpyxl.styles import Font,PatternFill,Alignment
from openpyxl.worksheet.datavalidation import DataValidation

ROOT=Path(__file__).resolve().parents[1];OLD=ROOT.parent/'v2_300篇';V3=ROOT.parent/'v3_期刊格式修订_20261028'
spec=importlib.util.spec_from_file_location('downloader',Path(__file__).with_name('download_references.py'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
lib=json.loads((OLD/'library_300.json').read_text(encoding='utf-8'));new={r['number']:r for r in json.loads((V3/'library_v3.json').read_text(encoding='utf-8'))}

# An ordinary journal download required the same session/permission check used by its public button.
for n in [2,4,31]:
    candidate=ROOT/'records'/f'{n:03d}_candidate.pdf'
    if candidate.exists() and candidate.read_bytes()[:5]==b'%PDF-':
        row=next(r for r in lib if r['number']==n);record=json.loads((ROOT/'records'/f'{n:03d}.json').read_text(encoding='utf-8'))
        target=ROOT/'PDF'/module.filename(row);shutil.copy2(candidate,target)
        url={4:'https://www.joca.cn/CN/article/downloadArticleFile.do?attachType=PDF&id=26336',31:'https://link.springer.com/content/pdf/10.1007/978-3-319-46448-0_27.pdf'}.get(n,row['url'])
        record.update(status='downloaded',file=str(target.relative_to(ROOT)),bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),download_url=url,requested_url=url,version='出版方公开全文' if n in [4,31] else '作者公开稿',access_date='2026-10-28')
        record=module.validate(record);(ROOT/'records'/f'{n:03d}.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
records=[]
for row in lib:
    record=json.loads((ROOT/'records'/f"{row['number']:03d}.json").read_text(encoding='utf-8'))
    n=row['number'];record['current_doi']=new[n].get('doi','');record['citation_v3']=new[n]['citation']
    if n==36 and record['status']=='downloaded':record['identity_check']='已目视核对首页题名和作者；扫描作者稿，页数与出版版不同';record['version']='作者公开稿（扫描版，首页标注in press 1998）'
    if n==54:record['identity_check']='首页与作者一致；PDF题名含Soft-thresold拼写，会议目录为Soft-Threshold'
    if n==6 and record['status']!='downloaded':record['unresolved_note']='出版方入口返回403；未找到可下载的公开作者稿，可通过学校图书馆或文献传递获取'
    elif record['status']!='downloaded':record['unresolved_note']='公开入口本轮未成功取得PDF，详见尝试日志；未用其他论文代替'
    records.append(record)

headers=['原编号','题名','年份','作者—年份引用','下载状态','PDF','版本','页数','题名检查','DOI','分类','阅读顺序','人工核验','作者年份是否一致','引文是否支持正文','核验页码','备注','官方入口','下载来源','SHA256']
rows=[]
for r in records:
    rows.append([r['number'],r['title'],r['year'],r['citation_v3'],'已下载' if r['status']=='downloaded' else '未下载',r.get('file',''),r.get('version',''),r.get('pages',''),r.get('identity_check',''),r['current_doi'],r['category'],r['priority'],'未核验','','','','',r['original_source'],r.get('download_url',''),r.get('sha256','')])
with (ROOT/'01_300篇下载与核验清单.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.writer(f);w.writerow(headers);w.writerows(rows)
wb=Workbook();ws=wb.active;ws.title='300篇逐条核验';ws.append(headers)
for row in rows:
    ws.append(row);i=ws.max_row
    if row[5]:ws.cell(i,6).hyperlink=row[5].replace('\\','/');ws.cell(i,6).font=Font(color='0563C1',underline='single')
    for col in [18,19]:
        if ws.cell(i,col).value:ws.cell(i,col).hyperlink=ws.cell(i,col).value;ws.cell(i,col).font=Font(color='0563C1')
    ws.row_dimensions[i].height=44
ws.freeze_panes='E2';ws.auto_filter.ref=ws.dimensions
for c in ws[1]:c.font=Font(bold=True,color='FFFFFF');c.fill=PatternFill('solid',fgColor='25495C')
widths=[8,60,8,30,12,36,36,8,40,32,26,18,14,20,20,16,44,30,30,30]
from openpyxl.utils import get_column_letter
for i,w in enumerate(widths,1):ws.column_dimensions[get_column_letter(i)].width=w
for row in ws.iter_rows(min_row=2):
    for c in row:c.alignment=Alignment(vertical='top',wrap_text=True)
dv=DataValidation(type='list',formula1='"未核验,核验中,已核验,需复查"');ws.add_data_validation(dv);dv.add('M2:M301')
miss=wb.create_sheet('未下载条目');miss.append(['原编号','题名','官方入口','原因'])
for r in records:
    if r['status']!='downloaded':miss.append([r['number'],r['title'],r['original_source'],r['unresolved_note']])
for k,v in {'A':10,'B':65,'C':70,'D':80}.items():miss.column_dimensions[k].width=v
meta=wb.create_sheet('使用说明');meta.append(['事项','说明'])
for x in [('原编号','对应v2阅读稿编号；v3改为作者—年份引用，仍保留对照。'),('核验','已下载及题名匹配不等于全文精读，也不等于正文论断已获证实。'),('版本','作者稿、预印本与会议/期刊版的页码、表号可能不同，应对照说明。'),('DOI','使用v3补查结果；空白表示未核定，不能推断论文无DOI。'),('如何使用','在PDF列点链接打开全文；人工核验列可选状态，记录对应页码与勘误。')]:meta.append(x)
meta.column_dimensions['A'].width=15;meta.column_dimensions['B'].width=110
wb.save(ROOT/'01_300篇下载与核验清单.xlsx')

good=[r for r in records if r['status']=='downloaded'];missing=[r for r in records if r['status']!='downloaded'];hashes=collections.defaultdict(list)
for r in good:hashes[r['sha256']].append(r['number'])
stats={'total':300,'downloaded':len(good),'missing':[r['number'] for r in missing],'bytes':sum(r['bytes'] for r in good),'duplicate_hash_groups':[v for v in hashes.values() if len(v)>1]}
(ROOT/'download_summary.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
(ROOT/'manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
rows_html=[]
for r in records:
    pdf='<a href="'+quote(r['file'].replace('\\','/'),safe='/')+'">打开PDF</a>' if r.get('status')=='downloaded' else '未取得PDF'
    rows_html.append('<tr>'+''.join('<td>'+v+'</td>' for v in [str(r['number']).zfill(3),html.escape(r['title']),str(r['year']),pdf,html.escape(r.get('version','')),html.escape(r['citation_v3']),'<a href="'+html.escape(r['original_source'],quote=True)+'">官方入口</a>'])+'</tr>')
page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>300篇文献核验入口</title><style>body{{font-family:"Microsoft YaHei",sans-serif;margin:30px;color:#203342}}h1{{font-size:24px}}table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #dce2e6;padding:10px;text-align:left;vertical-align:top}}input{{width:70%;padding:10px;margin:15px 0}}a{{color:#17678a}}thead{{background:#eef3f5}}</style><h1>300篇文献核验入口</h1><p>已下载 {len(good)} / 300 篇。文件编号沿用v2；v3引用见作者—年份列。下载和题名检查不代表完成全文核验。</p><p><a href="01_300篇下载与核验清单.xlsx">打开Excel核验清单</a></p><input id="q" placeholder="筛选编号、题名、年份或作者"><table><thead><tr><th>编号</th><th>题名</th><th>年份</th><th>PDF</th><th>版本</th><th>v3引用</th><th>来源</th></tr></thead><tbody>{''.join(rows_html)}</tbody></table><script>document.getElementById('q').oninput=function(){{const q=this.value.toLowerCase();document.querySelectorAll('tbody tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(q))}}</script></html>'''
(ROOT/'00_文献核验入口.html').write_text(page,encoding='utf-8')
readme=f'# 300篇参考文献全文与核验入口\n\n截至2026年10月28日，已取得{len(good)}篇PDF，共{stats["bytes"]/1024**3:.2f} GiB。未取得全文的编号：'+('、'.join(str(r['number']) for r in missing) or '无')+'。\n\n打开 `00_文献核验入口.html` 可检索题名并打开PDF；打开 `01_300篇下载与核验清单.xlsx` 可逐条记录核验状态、对应页码和勘误。PDF文件名以v2原编号开头，作者—年份列对应v3。\n\n公开作者稿和预印本已标明版本；PDF页数与期刊印刷页码可能不同。下载成功、文件可解析和首页题名匹配不等于论文全文已读，更不等于正文引用已经逐句验证。\n\n未下载条目：\n\n'+''.join(f'- [{r["number"]}] {r["title"]}：{r["unresolved_note"]}。[官方入口]({r["original_source"]})\n' for r in missing)
(ROOT/'00_下载说明.md').write_text(readme,encoding='utf-8')
print(json.dumps(stats,ensure_ascii=False,indent=2))
