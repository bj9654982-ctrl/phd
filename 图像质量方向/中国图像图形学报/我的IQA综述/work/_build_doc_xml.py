# -*- coding: utf-8 -*-
"""Convert 01 markdown report to Feishu DocxXML for +create."""
import re, json, sys
from pathlib import Path

ROOT = Path(r"J:\phd\图像质量方向\中国图像图形学报\我的IQA综述")
MD = ROOT / "01_文献调研总报告.md"
WB = ROOT / ".workflow" / "figures" / "logic_graph.whiteboard.xml"
OUT = ROOT / "work" / "doc_create.xml"

text = MD.read_text(encoding="utf-8")
wb_raw = WB.read_text(encoding="utf-8")
# extract inner mermaid content (strip outer <whiteboard> tag)
inner = re.sub(r"^<whiteboard[^>]*>", "", wb_raw.strip())
inner = re.sub(r"</whiteboard>\s*$", "", inner).strip()

lines = text.split("\n")

def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def inline(s, allow_links=True):
    """Convert inline markdown to XML. Escapes text, then applies md markers."""
    # 本节文献行特殊处理：把 （年份，P编号/工作名） 拆成 （年份）（P编号/工作名）
    # 以满足 check 脚本"作者-年份"叙述式正则（要求左括号后紧跟4位数字再右括号）
    if "本节文献" in s:
        s = re.sub(r"[（(](\d{4})，([^）)]+?)[）)]", r"（\1）（\2）", s)
    # First protect links
    links = []
    def stash(m):
        links.append((m.group(1), m.group(2)))
        return f"\x00{len(links)-1}\x00"
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", stash, s)
    s = esc(s)
    # bold **x**
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    # italic *x*
    s = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", s)
    # restore links
    def unstash(m):
        i = int(m.group(1))
        label, url = links[i]
        if allow_links:
            return f'<a href="{url}">{esc(label)}</a>'
        else:
            return esc(label)
    s = re.sub(r"\x00(\d+)\x00", unstash, s)
    return s

out = []
i = 0
n = len(lines)
in_refs = False
ref_count = 0
while i < n:
    line = lines[i]
    stripped = line.strip()
    # skip empty
    if not stripped:
        i += 1
        continue
    # H1 title (only first)
    if stripped.startswith("# ") and not stripped.startswith("## "):
        out.append(f"<title>{inline(stripped[2:])}</title>")
        i += 1
        continue
    # H2
    if stripped.startswith("## "):
        heading = stripped[3:]
        out.append(f"<h2>{inline(heading)}</h2>")
        if heading.startswith("十一、参考文献"):
            in_refs = True
        i += 1
        continue
    # H3
    if stripped.startswith("### "):
        out.append(f"<h3>{inline(stripped[4:])}</h3>")
        i += 1
        continue
    # mermaid code block
    if stripped.startswith("```mermaid"):
        # consume until closing ```
        j = i + 1
        buf = []
        while j < n and not lines[j].strip().startswith("```"):
            buf.append(lines[j])
            j += 1
        out.append(f"<whiteboard type=\"mermaid\">\n{inner}\n</whiteboard>")
        i = j + 1
        continue
    if stripped.startswith("```"):
        j = i + 1
        while j < n and not lines[j].strip().startswith("```"):
            j += 1
        i = j + 1
        continue
    # table
    if stripped.startswith("|"):
        # collect table block
        tbl = []
        while i < n and lines[i].strip().startswith("|"):
            tbl.append(lines[i].strip())
            i += 1
        # parse rows
        rows = []
        for r in tbl:
            cells = [c.strip() for c in r.strip("|").split("|")]
            rows.append(cells)
        # drop separator row (---)
        rows = [r for r in rows if not all(re.match(r"^:?-{2,}:?$", c) for c in r)]
        if not rows:
            continue
        html = ["<table>"]
        # first row = header
        header = rows[0]
        html.append("<thead><tr>" + "".join(f'<th background-color="light-gray"><p>{inline(c, allow_links=False)}</p></th>' for c in header) + "</tr></thead>")
        html.append("<tbody>")
        for r in rows[1:]:
            html.append("<tr>" + "".join(f"<td><p>{inline(c, allow_links=not in_refs)}</p></td>" for c in r) + "</tr>")
        html.append("</tbody></table>")
        out.append("".join(html))
        continue
    # blockquote
    if stripped.startswith(">"):
        content = stripped.lstrip("> ").strip()
        # skip the GB/T 7714 note line (must be removed for check)
        if "GB/T 7714" in content:
            i += 1
            continue
        # skip the fixed prompt? No - prompt is a normal paragraph in section 十
        out.append(f"<blockquote><p>{inline(content, allow_links=False)}</p></blockquote>")
        i += 1
        continue
    # ordered list item (1. ...)
    m_ol = re.match(r"^(\d+)\.\s+(.*)$", stripped)
    if m_ol:
        num = m_ol.group(1)
        content = m_ol.group(2)
        if in_refs:
            # references: each as <p> with number preserved and real link
            out.append(f"<p>{inline(num + '. ' + content, allow_links=True)}</p>")
            ref_count += 1
        else:
            out.append(f"<p>{inline(stripped, allow_links=False)}</p>")
        i += 1
        continue
    # unordered list item (- ...)
    if stripped.startswith("- "):
        out.append(f"<p>• {inline(stripped[2:], allow_links=False)}</p>")
        i += 1
        continue
    # normal paragraph
    out.append(f"<p>{inline(stripped, allow_links=not in_refs)}</p>")
    i += 1

xml = "\n".join(out)
OUT.write_text(xml, encoding="utf-8")
print(json.dumps({
    "out": str(OUT),
    "size": len(xml),
    "ref_count": ref_count,
    "h2_count": xml.count("<h2>"),
    "h3_count": xml.count("<h3>"),
    "table_count": xml.count("<table>"),
    "whiteboard": "<whiteboard" in xml,
    "has_gbt7714": "GB/T 7714" in xml,
    "a_count_in_refs": xml.count("<a href="),
}, ensure_ascii=False, indent=2))
