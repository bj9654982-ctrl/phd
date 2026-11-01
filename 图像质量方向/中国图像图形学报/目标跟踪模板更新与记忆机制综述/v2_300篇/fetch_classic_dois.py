from pathlib import Path
import requests, json, time, sys
sys.stdout.reconfigure(encoding='utf-8')
root=Path(__file__).resolve().parent
dois=[
('C1','10.1023/A:1008078328650',1998),
('C2','10.1109/TPAMI.2003.1195991',2003),
('C3','10.1007/s11263-007-0075-7',2008),
('C4','10.1109/TPAMI.2010.226',2011),
('C5','10.1109/TPAMI.2011.239',2012),
('C6','10.1109/ICCV.2011.6126251',2011),
('C7','10.1007/978-3-642-33765-9_50',2012),
('C8','10.1007/978-3-319-10599-4_13',2014),
('C9','10.5244/C.28.65',2014),
('C10','10.1007/978-3-319-46454-1_29',2016),
]
path=root/'classic_doi_metadata.json'
out=json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
for key,doi,year in dois:
 if any(r['key']==key and 'record' in r for r in out): continue
 try:
  r=requests.get('https://api.crossref.org/works/'+doi,timeout=(8,20));r.raise_for_status()
  v={'key':key,'doi':doi,'year':year,'record':r.json()['message']}
  out=[x for x in out if x['key']!=key]+[v]
  print(key,v['record'].get('title'),v['record'].get('page'),flush=True)
 except Exception as e:print(key,str(e),flush=True)
 path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
 time.sleep(1.5)
