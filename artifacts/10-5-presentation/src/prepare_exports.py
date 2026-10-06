"""Copy the final PDF and derive overlay notes and labels for its PPTX mirror."""
from pathlib import Path
import json
import re
import shutil
from pypdf import PdfReader

BASE = Path(__file__).resolve().parents[1]
text = (BASE / "src/main.tex").read_text()
notes = re.findall(r"\\note\{([^{}]*)\}", text)
counts = [1,4,2,2,4,2,1,1,1]
reader = PdfReader(BASE / "build/main.pdf")
assert len(reader.pages) == sum(counts) == 18
assert len(notes) == len(counts)
pages=[]
n=0
for frame,count in enumerate(counts):
    for step in range(1,count+1):
        page=reader.pages[n]
        pages.append({"page":n+1,"frame":frame,"step":step,
                      "notes":notes[frame],"text":page.extract_text() or ""})
        n+=1
(BASE / "build/page-metadata.json").write_text(json.dumps(pages,indent=2)+"\n")
shutil.copyfile(BASE / "build/main.pdf", BASE / "presentation.pdf")
print("Prepared 18 Beamer pages, speaker notes, and final PDF")
