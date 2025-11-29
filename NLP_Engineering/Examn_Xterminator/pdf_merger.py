#!/usr/bin/env python3
"""
pdf_merger.py

Fusiona PDFs ordenándolos por fecha (metadata /CreationDate o fecha de modificación del fichero).

Argumentos por defecto solicitados:
    --indir default "./"
    --out default "./"
    --order default "desc"
    --key default "pdfdate"
    --recursive flag
    --dry-run flag

Ejemplo:
    python pdf_merger.py --indir "F:/mis_pdfs" --out "./merged.pdf"
    python pdf_merger.py file1.pdf file2.pdf --out merged.pdf
    python pdf_merger.py --indir . --dry-run
"""
import argparse
from pathlib import Path
from datetime import datetime
import sys
import re

# Intentar importar pypdf; si no está, imprimir instrucción clara
try:
    from pypdf import PdfReader, PdfWriter
except Exception:
    print("Este script requiere 'pypdf'. Instálalo: python -m pip install pypdf", file=sys.stderr)
    raise

PDF_CREATION_RE = re.compile(r"D:(\d{4})(\d{2})?(\d{2})?(\d{2})?(\d{2})?(\d{2})?")

def parse_pdf_creation_date(raw: str):
    if not raw:
        return None
    m = PDF_CREATION_RE.search(str(raw))
    if not m:
        return None
    year = int(m.group(1))
    month = int(m.group(2) or 1)
    day = int(m.group(3) or 1)
    hour = int(m.group(4) or 0)
    minute = int(m.group(5) or 0)
    second = int(m.group(6) or 0)
    try:
        return datetime(year, month, day, hour, minute, second)
    except Exception:
        return None

def get_pdf_date(path: Path):
    """Devuelve (datetime, 'pdfmeta' | 'mtime' | 'unknown')"""
    try:
        reader = PdfReader(str(path))
        meta = None
        try:
            meta = reader.metadata
        except Exception:
            meta = getattr(reader, "documentInfo", None)
        creation_raw = None
        if meta:
            # pypdf metadata puede exponer .creation_date o keys
            if hasattr(meta, "creation_date") and meta.creation_date:
                creation_raw = meta.creation_date
            else:
                # probar keys comunes
                keys = ("/CreationDate", "CreationDate", "creation_date", "/ModDate", "ModDate")
                if isinstance(meta, dict):
                    for k in keys:
                        if k in meta and meta[k]:
                            creation_raw = meta[k]; break
                else:
                    for k in keys:
                        if hasattr(meta, k):
                            v = getattr(meta, k)
                            if v:
                                creation_raw = v; break
        if creation_raw:
            dt = parse_pdf_creation_date(creation_raw)
            if dt:
                return dt, "pdfmeta"
    except Exception:
        # lector puede fallar con PDFs corruptos o versiones distintas -> fallback
        pass

    # fallback a mtime
    try:
        ts = path.stat().st_mtime
        return datetime.fromtimestamp(ts), "mtime"
    except Exception:
        return datetime.fromtimestamp(0), "unknown"

def collect_input_files(inputs, indir, recursive):
    files = []
    if inputs:
        for p in inputs:
            files.append(Path(p))
    elif indir:
        base = Path(indir)
        if not base.exists():
            raise FileNotFoundError(f"Input directory not found: {indir}")
        if recursive:
            files = [p for p in base.rglob("*.pdf") if p.is_file()]
        else:
            files = [p for p in base.glob("*.pdf") if p.is_file()]
    else:
        raise ValueError("No inputs provided.")
    # filtrar normalizar
    res = []
    for f in files:
        if not f.exists():
            print(f"Warning: file not found and will be skipped: {f}", file=sys.stderr)
            continue
        if f.suffix.lower() != ".pdf":
            print(f"Skipping non-pdf: {f}", file=sys.stderr)
            continue
        res.append(f.resolve())
    return res

def append_pdf_to_writer(path: Path, writer: PdfWriter):
    try:
        reader = PdfReader(str(path))
        if getattr(reader, "is_encrypted", False):
            try:
                # intento con contraseña vacía
                reader.decrypt("") 
            except Exception:
                print(f"Warning: encrypted PDF skipped (needs password): {path}", file=sys.stderr)
                return False
        for p in reader.pages:
            writer.add_page(p)
        return True
    except Exception as e:
        print(f"Error reading {path}: {e}", file=sys.stderr)
        return False

def merge(sorted_files, out_file: Path, dry_run=False):
    if dry_run:
        print("Dry run. The following order would be merged:")
        for f in sorted_files:
            print(" ", f)
        return
    writer = PdfWriter()
    appended = 0
    for p in sorted_files:
        ok = append_pdf_to_writer(p, writer)
        if ok:
            appended += 1
            print(f"Appended: {p}")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "wb") as fh:
        writer.write(fh)
    print(f"\nMerged {appended} files -> {out_file}")

def main():
    parser = argparse.ArgumentParser(description="Merge PDFs ordering by date (pdf metadata or file mtime).")
    parser.add_argument("inputs", nargs="*", help="PDF files to merge (if omitted, use --indir)")
    parser.add_argument("--indir", "-d", default="./", help="Directory containing PDFs to merge (alternative to listing files).")
    parser.add_argument("--out", "-o", default="./", help="Output merged PDF path.")
    parser.add_argument("--recursive", "-r", action="store_true", help="If --indir, search recursively.")
    parser.add_argument("--order", choices=("asc","desc"), default="desc", help="Sort order by date. desc=newest first (default).")
    parser.add_argument("--key", choices=("pdfdate","mtime"), default="pdfdate", help="Prefer date source: 'pdfdate' (default) or file 'mtime'.")
    parser.add_argument("--dry-run", action="store_true", help="Show order without merging.")
    args = parser.parse_args()

    try:
        files = collect_input_files(args.inputs, args.indir, args.recursive)
    except Exception as e:
        print("Error collecting input files:", e, file=sys.stderr)
        sys.exit(2)

    if not files:
        print("No PDF files found to merge.", file=sys.stderr)
        sys.exit(1)

    # construir lista con (path, date, source)
    file_infos = []
    for f in files:
        dt, src = get_pdf_date(f)
        if args.key == "mtime":
            dt = datetime.fromtimestamp(f.stat().st_mtime)
            src = "mtime"
        file_infos.append((f, dt, src))

    reverse = (args.order == "desc")
    file_infos.sort(key=lambda t: (t[1], str(t[0]).lower()), reverse=reverse)

    print("Files to merge in the following order (date, source):")
    for p, dt, src in file_infos:
        print(f"  {dt.isoformat()}  [{src}]  {p}")

    sorted_files = [t[0] for t in file_infos]

    # manejar argumento --out por defecto './' -> si es directorio usar merged_{timestamp}.pdf
    out_arg = Path(args.out)
    if out_arg.exists() and out_arg.is_dir():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_arg / f"merged_{timestamp}.pdf"
    else:
        # si out termina en separador o es "./" (que puede no existir), tratar como directorio
        if str(args.out).endswith(("/", "\\")) or args.out in ("./", ".\\"):
            out_dir = Path(args.out)
            out_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = out_dir / f"merged_{timestamp}.pdf"
        else:
            # si la ruta dada tiene extensión .pdf la usamos; si no, la interpretamos como dir
            if out_arg.suffix.lower() == ".pdf":
                out_path = out_arg
            else:
                # tratar como directorio
                out_dir = out_arg
                out_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = out_dir / f"merged_{timestamp}.pdf"

    merge(sorted_files, out_path, dry_run=args.dry_run)

if __name__ == "__main__":
    main()
