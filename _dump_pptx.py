import re
import sys
import zipfile
from xml.etree import ElementTree as ET

def dump(path):
    z = zipfile.ZipFile(path)
    names = [n for n in z.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", n)]
    names.sort(key=lambda n: int(re.search(r"(\d+)", n.split("/")[-1]).group(1)))
    print("FILE", path, "N", len(names))
    for n in names:
        root = ET.fromstring(z.read(n))
        texts = []
        for t in root.iter("{http://schemas.openxmlformats.org/drawingml/2006/main}t"):
            if t.text and t.text.strip():
                texts.append(t.text.strip())
        print("---", n.split("/")[-1], "---")
        print(" | ".join(texts))
        note = n.replace("slides/slide", "notesSlides/notesSlide")
        if note in z.namelist():
            nr = ET.fromstring(z.read(note))
            nt = [
                t.text.strip()
                for t in nr.iter("{http://schemas.openxmlformats.org/drawingml/2006/main}t")
                if t.text and t.text.strip()
            ]
            print("NOTES:", " ".join(nt))
        print()

sys.stdout.reconfigure(encoding="utf-8")
dump(sys.argv[1])
