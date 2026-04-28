from pathlib import Path
import PyPDF2

p = Path(r"c:/Users/Asus/Desktop/U/Semillero/Guia para ejecutar en local LSTM lenguaje de señas.pdf")
reader = PyPDF2.PdfReader(p)
print('pages', len(reader.pages))
for i, page in enumerate(reader.pages, start=1):
    txt = page.extract_text() or ''
    print('\n--- PAGE', i, '---\n')
    print(txt)
