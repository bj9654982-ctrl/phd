# -*- coding: utf-8 -*-
import subprocess, json, sys, re
from pathlib import Path

DOC_ID = "KlWldFHApo23RQxa5gFcliMBn6c"
OUT = Path(r"J:\phd\图像质量方向\中国图像图形学报\我的IQA综述\.workflow\doc_fetch.xml")
CWD = r"J:\phd\图像质量方向\中国图像图形学报\我的IQA综述\work"

r = subprocess.run(
    ["lark-cli", "docs", "+fetch", "--doc", DOC_ID, "--doc-format", "xml", "--detail", "full", "--format", "json"],
    cwd=CWD, capture_output=True, text=True, encoding="utf-8"
)
if r.returncode != 0:
    print("STDERR:", r.stderr[:2000]); sys.exit(1)
data = json.loads(r.stdout)

def find_xml(obj):
    if isinstance(obj, str) and ("<title" in obj or "<h2" in obj or "<p>" in obj):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            x = find_xml(v)
            if x: return x
    if isinstance(obj, list):
        for v in obj:
            x = find_xml(v)
            if x: return x
    return None

xml = find_xml(data)
# Restore &gt; inside whiteboard blocks so check script's "-->|label|" regex can match
def fix_wb(m):
    inner = m.group(0)
    inner = inner.replace("&gt;", ">").replace("&lt;", "<").replace("&amp;", "&")
    return inner
xml = re.sub(r"<whiteboard\b[^>]*>.*?</whiteboard>", fix_wb, xml, flags=re.DOTALL)

OUT.write_text(xml, encoding="utf-8")
# quick stats
wb = re.search(r"<whiteboard\b[^>]*>.*?</whiteboard>", xml, re.DOTALL)
edge_labels = re.findall(r"-->\|[^|]+?\|", wb.group(0)) if wb else []
lineage = re.findall(r"\bN\d+\b.*?-->\|[^|]+?\|.*?\bN\d+\b", wb.group(0), re.DOTALL) if wb else []
print(json.dumps({
    "len": len(xml),
    "whiteboard_found": bool(wb),
    "edge_label_count": len(edge_labels),
    "sample_edges": edge_labels[:5],
    "本节文献_count": xml.count("本节文献"),
}, ensure_ascii=False, indent=2))
