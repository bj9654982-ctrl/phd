from pathlib import Path
import re, json
import fitz
from docx import Document

root = Path(__file__).resolve().parent
stem = "03_综述完整初稿_v1_20261025"
text = (root / (stem + ".md")).read_text(encoding="utf-8")
body = text.split("## 参考文献")[0]
cited = {int(n) for group in re.findall(r"\[([0-9,]+)\]", body) for n in group.split(",")}
refs = {int(n) for n in re.findall(r"^\[(\d+)\]", text, re.M)}
pictures = re.findall(r"^!\[.*?\]\((.*?)\)", text, re.M)
doc = Document(root / (stem + ".docx"))
with fitz.open(root / (stem + ".pdf")) as pdf:
    result = {
        "missing_references": sorted(cited - refs),
        "uncited_references": sorted(refs - cited),
        "images_exist": all(Path(p).exists() for p in pictures),
        "placeholder_count": len(re.findall(r"【待|待测|待核实", text)),
        "docx_tables": len(doc.tables),
        "docx_images": len(doc.inline_shapes),
        "pdf_pages": len(pdf),
        "pdf_bookmarks": len(pdf.get_toc()),
        "table8_pages": [i + 1 for i, page in enumerate(pdf) if "表8 HIPTrack" in page.get_text()],
    }
    for page in pdf:
        if "表8 HIPTrack" in page.get_text():
            page.get_pixmap(matrix=fitz.Matrix(1.3, 1.3)).save(root / "preview_table8.png")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    (root / "integrity_check.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    assert not result["missing_references"]
    assert result["images_exist"]
    assert result["placeholder_count"] == 0
    assert result["docx_tables"] == 9
