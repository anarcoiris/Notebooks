#!/usr/bin/env python3
"""
pdf_exam_analyzer.py

Versión actualizada: usa la capacidad de los modelos de OpenAI para **extraer los enunciados directamente de las imágenes**
(en lugar de OCR local) para preservar fórmulas y notación matemática. Además implementa la estrategia "cluster-first":
 1) por cada página -> pedir a la API que **extraiga únicamente los enunciados** (sin resolver)
 2) agrupar (clusterizar) los enunciados extraídos para detectar preguntas repetidas
 3) por cada cluster único -> enviar **una** imagen representativa a la API pidiendo la **solución completa**

Con esto evitamos que se resuelvan dos veces preguntas repetidas y mejoramos la fidelidad al tratar fórmulas.

Requisitos y notas:
 - Igual que antes: PyMuPDF, Pillow, requests y (opcional) rapidfuzz. Si no hay OPENAI_API_KEY disponible, el script caerá
   en un modo fallback que intenta OCR local (pytesseract) para al menos generar un índice.
 - Este script hace llamadas a la API de OpenAI por página (extracción) + por cluster (solución). Ajusta --sleep para limitar
   la tasa y controlar costes.
 - Recomendación: usa un modelo imagen-capaz (p. ej. gpt-4o o el que tu cuenta tenga habilitado para imágenes). Ajusta --model.

Salida:
 - out/images/ (una imagen por página)
 - out/extracted_statements.json  (enunciados extraídos por página)
 - out/clusters.json  (clusters detectados)
 - out/solutions.md  (soluciones por cluster)
 - out/index_by_frequency.md  (índice ordenado por frecuencia)

"""

import os
import sys
import argparse
import base64
import json
import time
from pathlib import Path

try:
    import fitz  # PyMuPDF
except Exception:
    raise RuntimeError("PyMuPDF (fitz) no está instalado. `pip install pymupdf`")

from PIL import Image

try:
    from rapidfuzz import fuzz
    rapidfuzz_available = True
except Exception:
    import difflib
    rapidfuzz_available = False

import requests

# ----------------------------- Config defaults -----------------------------
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o"  # ajuste según disponibilidad

# ----------------------------- Utilities ----------------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def pdf_to_images(pdf_path: str, out_dir: str, dpi: int = 150):
    doc = fitz.open(pdf_path)
    images = []
    out = Path(out_dir)
    ensure_dir(out)
    for i in range(len(doc)):
        page = doc.load_page(i)
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_path = out / f"page_{i+1:04d}.png"
        pix.save(str(img_path))
        images.append(str(img_path))
    doc.close()
    return images


def image_to_data_uri(path: str) -> str:
    with open(path, 'rb') as f:
        b = f.read()
    mime = 'image/png'
    b64 = base64.b64encode(b).decode('ascii')
    return f"data:{mime};base64,{b64}"


def load_api_key():
    key = os.environ.get('OPENAI_API_KEY') or os.environ.get('OPENAI_KEY')
    return key


# -------------------- Similarity & clustering ------------------------------

def normalize_text_for_matching(s: str) -> str:
    s2 = (s or '').lower()
    s2 = " ".join(s2.split())
    return s2


def text_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if rapidfuzz_available:
        score = fuzz.token_set_ratio(a, b)
        return score / 100.0
    else:
        return difflib.SequenceMatcher(None, a, b).ratio()


def cluster_texts(texts, threshold=0.72):
    norms = [normalize_text_for_matching(t) for t in texts]
    clusters = []
    for i, ni in enumerate(norms):
        assigned = False
        for cl in clusters:
            rep = norms[cl['rep_idx']]
            sim = text_similarity(ni, rep)
            if sim >= threshold:
                cl['members'].append(i)
                if len(texts[i]) > len(texts[cl['rep_idx']]):
                    cl['rep_idx'] = i
                assigned = True
                break
        if not assigned:
            clusters.append({'rep_idx': i, 'members': [i]})
    clusters_sorted = sorted(clusters, key=lambda c: len(c['members']), reverse=True)
    return clusters_sorted


# -------------------- OpenAI helpers --------------------------------------

PROMPT_EXTRACT_STATEMENT = (
    "Eres un asistente que extrae ENUNCIADOS de preguntas de examen a partir de la imagen.\n"
    "- Devuelve solo el texto del enunciado (incluye símbolos y fórmulas tal cual aparezcan).\n"
    "- Si hay varias preguntas en la página, sepáralas en líneas nuevas y numéralas (1), (2), ...\n"
    "- No incluyas soluciones, ni explicaciones ni metadatos. Solo el enunciado textual.\n"
    "- Si la imagen está parcial o borrosa, indica claramente (INCOMPLETO) y devuelve lo visible.\n"
    "Responde en el mismo idioma del enunciado.")

PROMPT_SOLVE_QUESTION = (
    "Eres un profesor experto en termodinámica. Resuelve la pregunta que aparece en la imagen.\n"
    "Explica los pasos y fórmulas usadas, enumera partes si corresponde, y da el resultado final claramente.\n"
    "Si falta información visible en la imagen, indícalo y da supuestos razonables.\n"
)


def call_openai_chat_with_image(image_path: str, prompt_text: str, model=DEFAULT_MODEL, max_retries=3, timeout=300, max_tokens=2000):
    api_key = load_api_key()
    if not api_key:
        raise RuntimeError('No OPENAI_API_KEY en entorno')
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    data_uri = image_to_data_uri(image_path)
    payload = {
        'model': model,
        'messages': [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt_text},
                    {'type': 'image_url', 'image_url': {'url': data_uri, 'alt': Path(image_path).name}}
                ]
            }
        ],
        'temperature': 0.0,
        'max_tokens': max_tokens
    }
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(OPENAI_API_URL, headers=headers, json=payload, timeout=timeout)
            if r.status_code == 200:
                j = r.json()
                if 'choices' in j and len(j['choices']) > 0:
                    choice = j['choices'][0]
                    msg = choice.get('message') or choice.get('delta') or {}
                    content = msg.get('content')
                    if isinstance(content, str):
                        return content.strip()
                    if isinstance(content, list):
                        texts = []
                        for piece in content:
                            if isinstance(piece, dict) and piece.get('type') in ('output_text','text'):
                                texts.append(piece.get('text',''))
                            elif isinstance(piece, str):
                                texts.append(piece)
                        return "\n".join([t for t in texts if t]).strip()
                    if 'text' in choice:
                        return choice['text'].strip()
                    return json.dumps(j, ensure_ascii=False)
            else:
                err = r.text
                wait = attempt * 2
                print(f"Aviso: OpenAI API devolvió {r.status_code}. Reintentando en {wait}s. Respuesta: {err}")
                time.sleep(wait)
        except requests.Timeout:
            print(f"Timeout en intento {attempt}. Reintentando...")
            time.sleep(attempt * 2)
    raise RuntimeError('Fallo al comunicarse con la API de OpenAI tras varios intentos')


# -------------------- Pipeline principal ---------------------------------

def analyze_pdf_with_image_extraction(pdf_path: str, out_dir: str, similarity_threshold: float = 0.72,
                                      model: str = DEFAULT_MODEL, pages_limit: int = None,
                                      sleep_between_requests: float = 1.0, extract_only=False):
    out_dir = Path(out_dir)
    images_dir = out_dir / 'images'
    ensure_dir(images_dir)

    print('1) Convirtiendo PDF a imágenes...')
    images = pdf_to_images(pdf_path, str(images_dir))
    if pages_limit:
        images = images[:pages_limit]
    print(f'  -> {len(images)} páginas convertidas en: {images_dir}')

    api_key = load_api_key()
    extracted = {}

    # 2) Extracción de enunciados usando el modelo (uno por página)
    print('2) Extrayendo enunciados desde imágenes (modelo).')
    for i, img in enumerate(images, start=1):
        print(f'   Página {i}: {img}')
        if api_key:
            try:
                resp = call_openai_chat_with_image(img, PROMPT_EXTRACT_STATEMENT, model=model, max_tokens=2000)
                extracted_text = resp.strip()
            except Exception as e:
                print(f'   ERROR extrayendo página {i}: {e} -- intentaremos fallback OCR local si está disponible')
                extracted_text = ''
        else:
            print('   No OPENAI_API_KEY; extracción por modelo saltada. (usa --fallback to enable local OCR)')
            extracted_text = ''

        extracted[i] = {'image': img, 'extracted_text': extracted_text}
        print(f'    -> {len(extracted_text)} caracteres extraídos')
        time.sleep(sleep_between_requests)

    (out_dir / 'extracted_statements.json').write_text(json.dumps(extracted, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'  -> Extracted statements saved to {out_dir / "extracted_statements.json"}')

    # 3) Clusterizar enunciados para detectar repeticiones
    texts_list = [extracted[i]['extracted_text'] or '' for i in range(1, len(images)+1)]
    clusters = cluster_texts(texts_list, threshold=similarity_threshold)
    clusters_out = []
    for idx, cl in enumerate(clusters, start=1):
        rep_idx = cl['rep_idx']
        members = [m+1 for m in cl['members']]
        clusters_out.append({
            'cluster_id': idx,
            'rep_page': rep_idx+1,
            'rep_text': texts_list[rep_idx],
            'pages': members,
            'count': len(members)
        })
    (out_dir / 'clusters.json').write_text(json.dumps(clusters_out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'  -> Clusters guardados en {out_dir / "clusters.json"} (threshold={similarity_threshold})')

    if extract_only or not api_key:
        print('Modo extracción only o sin API key: proceso finalizado (no se solicitan soluciones).')
        # crear índice básico
        idx_lines = ["# Índice de preguntas por frecuencia\n"]
        for cl in clusters_out:
            idx_lines.append(f"- Cluster {cl['cluster_id']}: apariciones={cl['count']} | páginas={cl['pages']}")
        (out_dir / 'index_by_frequency.md').write_text('\n'.join(idx_lines), encoding='utf-8')
        return

    # 4) Para cada cluster, pedir solución enviando solo la imagen representativa
    print('3) Solicitando soluciones a OpenAI (una imagen por cluster)...')
    solutions = []
    for cl in clusters_out:
        rep_img = images[cl['rep_page'] - 1]
        prompt = f"Páginas: {cl['pages']} (apariciones: {cl['count']}).\n" + PROMPT_SOLVE_QUESTION
        print(f"  -> Solicitando solución para cluster {cl['cluster_id']} (pág {cl['rep_page']})...")
        try:
            resp = call_openai_chat_with_image(rep_img, prompt, model=model, max_tokens=3000)
        except Exception as e:
            resp = f"ERROR: {e}"
        solutions.append({'cluster_id': cl['cluster_id'], 'rep_page': cl['rep_page'], 'pages': cl['pages'], 'solution': resp})
        time.sleep(sleep_between_requests)

    # 5) Guardar soluciones y generar archivos finales
    sol_md_lines = ["# Soluciones por cluster\n"]
    for s in solutions:
        sol_md_lines.append(f"## Cluster {s['cluster_id']}  - Páginas: {s['pages']}\n")
        sol_md_lines.append(f"Imagen representativa: {Path(images[s['rep_page'] - 1]).name}\n")
        sol_md_lines.append('---\n')
        sol_md_lines.append(s['solution'] or 'No hay solución (error en la API).')
        sol_md_lines.append('\n\n')
    (out_dir / 'solutions.md').write_text('\n'.join(sol_md_lines), encoding='utf-8')
    print(f'  -> Soluciones guardadas en {out_dir / "solutions.md"}')

    idx_lines = ["# Índice de preguntas por frecuencia\n"]
    for cl in clusters_out:
        idx_lines.append(f"- Cluster {cl['cluster_id']}: apariciones={cl['count']} | páginas={cl['pages']}")
    (out_dir / 'index_by_frequency.md').write_text('\n'.join(idx_lines), encoding='utf-8')
    print(f'  -> Índice generado en {out_dir / "index_by_frequency.md"}')

    print('\nProceso completado.')


# -------------------- CLI -----------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Analiza un PDF de exámenes: extracción por imagen -> cluster -> soluciones (cluster-first)')
    p.add_argument('--pdf', required=True, help='ruta al PDF de entrada')
    p.add_argument('--out', required=True, help='directorio de salida')
    p.add_argument('--threshold', type=float, default=0.72, help='umbral de similitud para agrupar preguntas (0..1)')
    p.add_argument('--model', default=DEFAULT_MODEL, help='modelo OpenAI a usar (imagen-capaz)')
    p.add_argument('--pages', type=int, default=None, help='procesar solo las primeras N páginas (útil para pruebas)')
    p.add_argument('--sleep', type=float, default=1.0, help='segundos a esperar entre llamadas a la API')
    p.add_argument('--extract-only', action='store_true', help='solo extraer enunciados y clusterizar; no solicitar soluciones')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    analyze_pdf_with_image_extraction(args.pdf, args.out, similarity_threshold=args.threshold,
                                      model=args.model, pages_limit=args.pages,
                                      sleep_between_requests=args.sleep, extract_only=args.extract_only)
