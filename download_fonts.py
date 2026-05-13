"""download_fonts.py — Descarga la fuente Orbitron para HORUS.

Ejecutar UNA SOLA VEZ desde la raíz del proyecto:
    python download_fonts.py
"""
import urllib.request
import os
from pathlib import Path

FONT_DIR = Path(__file__).parent / "assets" / "fonts"
FONT_DIR.mkdir(parents=True, exist_ok=True)

FONTS = [
    ("Orbitron-Regular.ttf",
     "https://github.com/google/fonts/raw/refs/heads/main/ofl/orbitron/static/Orbitron-Regular.ttf"),
    ("Orbitron-Bold.ttf",
     "https://github.com/google/fonts/raw/refs/heads/main/ofl/orbitron/static/Orbitron-Bold.ttf"),
    ("Orbitron-Medium.ttf",
     "https://github.com/google/fonts/raw/refs/heads/main/ofl/orbitron/static/Orbitron-Medium.ttf"),
]

print("Descargando fuentes Orbitron para HORUS...")
for name, url in FONTS:
    dest = FONT_DIR / name
    if dest.exists():
        print(f"  ✅ {name} ya existe — omitido")
        continue
    try:
        print(f"  ⬇  {name}...", end=" ", flush=True)
        urllib.request.urlretrieve(url, dest)
        print(f"OK ({dest.stat().st_size // 1024} KB)")
    except Exception as e:
        print(f"ERROR: {e}")

print("\nListo. Ejecuta main.py para jugar con la nueva fuente.")
