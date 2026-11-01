# -*- coding: utf-8 -*-
"""Assemble 01_文献调研总报告.md from synthesis_content.md + review_draft.md + final_library.json."""
import json, re
from pathlib import Path
WORK = Path(r"J:\phd\图像质量方向\中国图像图形学报\我的IQA综述\work")
PROJ = WORK.parent
wf = Path(r"J:\phd\图像质量方向\中国图像图形学报\我的IQA综述\.workflow")

syn = (WORK / "synthesis_content.md").read_text(encoding="utf-8")
review = (WORK / "review_draft.md").read_text(encoding="utf-8").strip()
wb = (wf / "figures" / "logic_graph.whiteboard.xml").read_text(encoding="utf-8")
lib = json.loads((WORK / "final_library.json").read_text(encoding="utf-8-sig"))["papers"]

# 1) strip handoff pool header (everything before first '# ' title)
m = re.search(r"^# ", syn, flags=re.M)
body = syn[m.start():]
# 2) cut appendix (review draft reproduced by synthesis) -> section 十 is formal
ap = re.search(r"^## 附：", body, flags=re.M)
if ap:
    body = body[:ap.start()].rstrip() + "\n"

# 3) replace section 五 placeholder text with the mermaid whiteboard
mermaid_match = re.search(r"<whiteboard[^>]*>\s*(.*?)\s*</whiteboard>", wb, flags=re.S)
mermaid_code = mermaid_match.group(1).strip()
sec5 = ("## 五、核心逻辑图\n\n"
        "下图以核心研究问题为根节点，按「参考信息可用性 × 建模范式」双轴串起 IQA 三十年研究脉络："
        "NSS→CNN→Transformer→CLIP/MLLM 的范式扩展与分化、传统 FR 指标在 AIGC 上的失效限定，以及 MLLM 微调派与未微调派之间的争论。\n\n"
        "```mermaid\n" + mermaid_code + "\n```\n")
body = re.sub(r"## 五、核心逻辑图\n.*?(?=^## 六、)", sec5, body, flags=re.S | re.M)

# 4) section 十 fixed prompt + review draft
sec10 = ("## 十、完整文献综述参考稿\n\n"
         "提示：以下内容是基于本次大方向调研生成的文献综述格式参考稿，用于示范如何组织已有研究、判断与引用；"
         "后续若要围绕具体细分题目写作，仍需根据该细分方向重新筛选、核验和补充文献，不应直接作为论文成品段落使用。\n\n"
         + review + "\n")
body = body.rstrip() + "\n\n" + sec10

# 5) references section, sorted by first-author surname
def sort_key(p):
    authors = p.get("authors") or ""
    if isinstance(authors, list):
        authors = "; ".join(str(a) for a in authors)
    first = authors.split(",")[0].split(";")[0].split(" and ")[0].strip()
    # western: surname is last token if in latin
    if re.search(r"[A-Za-z]", first):
        parts = first.split()
        surname = parts[-1] if parts else first
    else:
        surname = first
    return re.sub(r"[^A-Za-z\u4e00-\u9fff]", "", surname).lower()

papers = sorted(lib, key=sort_key)
lines = ["## 十一、参考文献\n",
         "> 著录格式为作者（年）著者-出版年制（GB/T 7714-2015 9.2），按第一作者姓氏字母/拼音顺序排列；正文引用为作者-年份夹注。"
         "C 级为 arXiv 预印本（条目中注明），终稿投稿前须替换为正式录用版或按期刊规则处理。\n"]
for i, p in enumerate(papers, 1):
    authors = p.get("authors","")
    if isinstance(authors, list):
        authors = "; ".join(str(a) for a in authors)
    authors = authors.strip()
    title = p.get("title","").strip()
    venue = p.get("venue","").strip()
    year = p.get("year","")
    url = (p.get("url") or "").strip()
    doi = (p.get("doi") or "").strip()
    tag = "（预印本）" if p.get("source_quality")=="C" and "arxiv" in url.lower() else ""
    entry = f"{authors}. {title}. {venue}, {year}.{tag}"
    link = url if url else (f"https://doi.org/{doi}" if doi else "")
    if link:
        entry += f" [原文/DOI]({link})"
    lines.append(f"{i}. {entry}")

full = body.rstrip() + "\n\n" + "\n".join(lines) + "\n"
(PROJ / "01_文献调研总报告.md").write_text(full, encoding="utf-8")
print("wrote 01_文献调研总报告.md")
print("chars:", len(full))
# sanity: no links in body before references
refpos = full.find("## 十一、参考文献")
print("body hyperlinks:", len(re.findall(r"\]\(https?://", full[:refpos])))
print("ref entries:", len(papers))
missing_link = [p["final_id"] for p in papers if not (p.get("url") or p.get("doi"))]
print("entries without link:", missing_link)
