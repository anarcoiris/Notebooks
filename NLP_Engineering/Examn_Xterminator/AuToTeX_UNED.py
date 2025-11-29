#!/usr/bin/env python3
"""
md_to_latex_simple.py

Conversor simplificado Markdown -> LaTeX adaptado al formato que indicas:
 - "## ..." -> \section*{...}
 - "### ..." -> \subsection*{...}
 - conserva display math ( $$..$$  y  \[..\] ) en su propio párrafo
 - conserva inline math ( \(..\) y $..$ ) sin tocar
 - convierte listas que comienzan con "- " en itemize
 - no introduce \text{} en ecuaciones ni transforma p_i -> \emph{i}
 - sólo escapa caracteres LaTeX en texto no-math

Requisitos: Python 3.8+
Uso: python md_to_latex_simple.py --md solutions.md --out solutions.tex --title "Soluciones"
"""
import re
import argparse
from pathlib import Path

PREAMBLE = r"""% Plantilla adaptada de:
%http://labfisicab1.blogspot.com/2013/09/plantilla-para-hacer-un-reporte-en-latex.html
%Plantilla creada el 8 de Octubre de 2019. Juan José Miralles Canals. UCLM-UNE
%jmiralles@albacete.uned.es


\documentclass[10pt,a4paper]{article}		%Paquete de Idioma y Tabla por cuadro
\usepackage[spanish,es-tabla]{babel}		%Codificación Alfabeto
\usepackage[utf8]{inputenc}					%Codificación de Fuente
\usepackage[T1]{fontenc}					%Índices y caratula estándar
\usepackage{makeidx}						%\makeindex	s
\usepackage{multirow}						%Tabla
\usepackage{graphicx}						%Gráficos
\usepackage{float} 							%Gráficos
%\usepackage{xcolor} 						%Gráficos
%\usepackage{color}							%color
\usepackage{xcolor} 						%color
\usepackage{amsmath}						%Matemática
\usepackage{amsfonts}						%Matemática
\usepackage{amssymb}						%Matemática
%\usepackage{amstext} 						%Matemática
\spanishdecimal{.} 							%quitar , anglosajona por . español
%\pagestyle{headings}						%Estilo de Página Numeración superior
\usepackage[pdftex]{hyperref}				%Hiperlinks
\usepackage{url}							%escibir urls \href{url}{text}
\usepackage{fancyhdr}						%Controla formato
\usepackage{listings}						%Incluir código fuente de LaTeX y Mathematica
\renewcommand{\lstlistingname}{Listado}		% Cambiamos Listing por Listado
%\usepackage{listingsutf8} 					% arregla acentos, no pirula
\usepackage{graphics}						%Mathematica
\usepackage{setspace}						%Mathematica
\usepackage[a4paper, left=25mm, right=25mm, top=30mm, bottom=20mm]{geometry}

\usepackage{hyperref}
\usepackage{pdfpages}
\newtheorem{theorem}{Teorema}[section]
\newtheorem{definition}{Definición}[section]
\newtheorem{example}{Ejemplo}[section]
\newtheorem{exercise}{Ejercicio}[section]
\newtheorem{lemma}{Lema}
\newtheorem{notation}{Nota}[section]
\newtheorem{problem}{Problema}[section]
\newtheorem{proposition}{Proposición}[section]
\newtheorem{solution}{Solución}[section]
\newtheorem{conclusion}{Conclusión}[section]
\newtheorem{summary}{Resumen}
\newtheorem{axiom}{Axioma}[section]
\newtheorem{conjecture}{Conjetura}
\newtheorem{corollary}{Corolario}[section]
\newtheorem{remark}{Comentario}

\newcommand{\n}{\mathbf}    %ATAJOS ÚTILES
\newcommand{\R}{\mathbb{R}}
\newcommand{\FF}{\mathbf{F}}
\newcommand{\AAA}{\mathbf{A}}
\newcommand{\BB}{\mathbf{B}}
\newcommand{\EE}{\mathbf{E}}
\newcommand{\HH}{\mathbf{H}}
\newcommand{\II}{\mathbf{I}}
\newcommand{\MM}{\mathbf{M}}
\newcommand{\mm}{\mathbf{m}}
\newcommand{\JJ}{\mathbf{J}}
\newcommand{\lol}{\frac{\mu_0}{4\pi}}
\newcommand{\KK}{\mathbf{K}}
\newcommand{\LL}{\mathbf{L}}
\newcommand{\PP}{\mathbf{P}}
\newcommand{\DD}{\mathbf{D}}
\newcommand{\SSe}{\mathbf{S}}
\newcommand{\ux}{\hat{\mathbf{u}}_x}
\newcommand{\uy}{\hat{\mathbf{u}}_y}
\newcommand{\uz}{\hat{\mathbf{u}}_z}
\newcommand{\up}{\hat{\mathbf{u}}_\rho}
\newcommand{\urho}{\hat{\mathbf{u}}_\rho}
\newcommand{\uphi}{\hat{\mathbf{u}}_\varphi}
\newcommand{\utheta}{\hat{\mathbf{u}}_\theta}
\newcommand{\ur}{\hat{\mathbf{u}}_r}
\newcommand{\rr}{\mathbf{r}}
\newcommand{\nn}{\mathbf{n}}
\newcommand{\ds}{d\mathbf{s}}
\newcommand{\dl}{d\mathbf{l}}
\newcommand{\dA}{d\mathbf{A}}
\newcommand{\vv}{\mathbf{v}}
\newcommand{\kk}{\mathbf{k}}
\newcommand{\zz}{\mathbf{z}}
\newcommand{\diver}{\nabla \cdot}
\newcommand{\rot}{\nabla \times}
\newcommand{\partialt}[1]{\frac{\partial #1}{\partial t}}
\newcommand{\partialtt}[1]{\frac{\partial^2 #1}{\partial t^2}}
\newcommand{\partialp}[1]{\frac{\partial #1}{\partial t}}
\newcommand{\partialz}[1]{\frac{\partial #1}{\partial z}}
\newcommand{\derivt}[1]{\frac{d#1}{dt}}




\newcommand{\mathsym}[1]{{}}				% MathematicaConversion
\newcommand{\unicode}[1]{{}}				% MathematicaConversion
\newcounter{mathematicapage}				% MathematicaConversion




%ruta de graficos

%Losficheros  gráficos se disponen en el subdirectorio imagenes, respecto del documento fuente.
%Ruta desde el documento fuente a los ficheros gráficos: imagenes/grafico.ext


% Incluir logos en la páginas en las escquinas superiores, tras la primera página

\lhead{\begin{picture}(0,0) \put(-60,-10){\includegraphics[width=20mm]{images/EscudoUned.jpg}} \end{picture}}
\rhead{\begin{picture}(0,0) \put(-2,0){\includegraphics[width=20mm]{images/ciencias.png}} \end{picture}}
\renewcommand{\headrulewidth}{0.5pt}
\pagestyle{fancy}

%%%%%%%%%%%%%%%% estilos listing %%%%%%%%%%%%%

\lstset{
        tabsize=2, % tab = 2 espacios
        backgroundcolor=\color[HTML]{F0F0F0}, % color de fondo
        captionpos=b, % posición de pie de código, b=debajo
        basicstyle=\scriptsize, % estilo de letra general
        frame=tb,
        captionpos=b,
        language=Mathematica,
        keepspaces=true,
        columns=fixed, % columnas alineadas
        extendedchars=true, % ASCII extendido
        breaklines=true, % partir líneas
        prebreak = \raisebox{0ex}[0ex][0ex]{\ensuremath{\hookleftarrow}}, % marcar final 																		%de línea con flecha
        showtabs=false, % no marcar tabulación
        showspaces=false, % no marcar espacios
        keywordstyle=\bfseries\color[HTML]{007020}, % estilo de palabras clave
        commentstyle=\itshape\color[HTML]{60A0B0}, % estilo de comentarios
        stringstyle=\color[HTML]{4070A0}, % estilo de strings
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






\begin{document}
\lstloadlanguages{Mathematica} %Cargar listing para Mathematica%

%Título
\title{__TITLE__}
\author{Santiago Javier Espino Heredero\\
Universidad Nacional de Educación y Distancia. (UNED)}

%%%%%%%%%%%%%%%%%%%%%%%%     Sustituir Cabecera  para memoria     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Título
%\title{Título de la memoria}


%Autor
%\author{Apellidos, Nombre \\ Facultad-Escuela\\
%\\ Departamento responsable de las prácticas\\ Centro Asociado (UNED) \\
% Laboratorio ...... \\
%N"o de práctica\\
%Grupo de práctica\\
%Nombre y apellidos del estudiante1\\ Nombre y apellidos del estudiante2\ \Nombre y apellidos del estudiante3 }

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


\maketitle{}  							%pone titulos para documento article
\tableofcontents						%dispone un indice
\pagebreak								%insertar salto de pagina
%Resumen

"""

ENDDOC = r"""
\end{document}
"""

# Regex for math spans (order matters: $$, \[ \], \( \), single $)
RE_DISPLAY_DOLLAR = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
RE_DISPLAY_BRACKET = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)
RE_INLINE_DOLLAR = re.compile(r"\$(.+?)\$", re.DOTALL)
RE_INLINE_PAREN = re.compile(r"\\\((.+?)\\\)", re.DOTALL)

# Combined pattern finder for masking (non-overlapping)
MATH_PATTERNS = [
    (re.compile(r"\$\$(.+?)\$\$", re.DOTALL), 'display'),
    (re.compile(r"\\\[(.+?)\\\]", re.DOTALL), 'display'),
    (re.compile(r"\$(.+?)\$", re.DOTALL), 'inline'),
    (re.compile(r"\\\((.+?)\\\)", re.DOTALL), 'inline'),
]

# Characters to escape in normal text (we DO NOT escape backslash '\' or '$' here)
LATEX_ESCAPE_CHARS = {
    '%': r'\%',
    '#': r'\#',
    '*': r'',
    '{': r'\{',
    '}': r'\}',
    '_': r'\_',   # we escape underscore outside math
    '&': r'\&',
    '^': r'\^{}',
    '~': r'\textasciitilde{}',
}

def escape_latex_text(s: str) -> str:
    """Escape a few special LaTeX characters in non-math text."""
    # Escape each char using a simple replace (order does not conflict)
    for ch, rep in LATEX_ESCAPE_CHARS.items():
        s = s.replace(ch, rep)
    return s

def find_math_spans(text: str):
    """Find non-overlapping math spans and return list of (start,end,raw,type)."""
    matches = []
    for patt, ptype in MATH_PATTERNS:
        for m in patt.finditer(text):
            matches.append((m.start(), m.end(), m.group(0), ptype))
    if not matches:
        return []
    matches.sort(key=lambda x: x[0])
    picked = []
    last_end = -1
    for start, end, raw, ptype in matches:
        if start >= last_end:
            picked.append((start, end, raw, ptype))
            last_end = end
    return picked

def mask_math_in_line(line: str):
    """Replace maths with tokens @@MATHi@@ and return masked line and list of spans."""
    spans = []
    picked = find_math_spans(line)
    if not picked:
        return line, spans
    out = []
    cur = 0
    for start, end, raw, ptype in picked:
        out.append(line[cur:start])
        token = f"@@MATH{len(spans)}@@"
        out.append(token)
        spans.append({'raw': raw, 'type': ptype})
        cur = end
    out.append(line[cur:])
    return ''.join(out), spans

def restore_math_tokens(text: str, spans):
    """Replace tokens @@MATHi@@ by original raw math strings. For display math, ensure blank lines around."""
    for i, s in enumerate(spans):
        token = f"@@MATH{i}@@"
        raw = s['raw']
        if s['type'] == 'display':
            replacement = '\n' + raw.strip() + '\n'
        else:
            replacement = raw
        text = text.replace(token, replacement)
    return text

def transform_text_preserving_math(text: str):
    """Escape text outside math tokens while preserving math tokens exactly."""
    masked, spans = mask_math_in_line(text)
    # split by tokens
    parts = re.split(r'(@@MATH\d+@@)', masked)
    out_parts = []
    token_rx = re.compile(r'@@MATH(\d+)@@')
    for p in parts:
        if p == '':
            continue
        m = token_rx.match(p)
        if m:
            out_parts.append(p)  # keep token as-is
        else:
            # this is normal text -> escape latex characters
            out_parts.append(escape_latex_text(p))
    combined = ''.join(out_parts)
    restored = restore_math_tokens(combined, spans)
    return restored

def convert_md_lines(lines):
    """Core converter: yields LaTeX lines."""
    out = []
    in_itemize = False
    in_code = False
    in_display_math_block = False
    display_delim = None  # '$$' or '\['

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].rstrip('\n')

        # code fence handling
        if re.match(r'^\s*```', line):
            if not in_code:
                in_code = True
                out.append('\\begin{verbatim}')
            else:
                in_code = False
                out.append('\\end{verbatim}')
            i += 1
            continue

        if in_code:
            out.append(line)
            i += 1
            continue

        # detect start of display math (either $$ or \[)
        strip = line.strip()
        start_display = False
        if strip.startswith('$$'):
            start_display = True
            display_delim = '$$'
        elif strip.startswith('\\['):
            start_display = True
            display_delim = '\\['

        if start_display:
            # collect full display block until closing delimiter
            buf = [line]
            if display_delim == '$$':
                # find closing $$ (could be same line)
                if strip.endswith('$$') and len(strip) > 2:
                    # single-line $$...$$
                    out.append('\n' + line.strip() + '\n')
                    i += 1
                    continue
                j = i + 1
                found = False
                while j < n:
                    buf.append(lines[j].rstrip('\n'))
                    if lines[j].strip().endswith('$$'):
                        found = True
                        break
                    j += 1
                out.append('\n' + '\n'.join(buf).strip() + '\n')
                i = j + 1
                continue
            else:
                # \[ ... \]
                if strip.endswith('\\]') and len(strip) > 2:
                    out.append('\n' + line.strip() + '\n')
                    i += 1
                    continue
                j = i + 1
                found = False
                while j < n:
                    buf.append(lines[j].rstrip('\n'))
                    if lines[j].strip().endswith('\\]'):
                        found = True
                        break
                    j += 1
                out.append('\n' + '\n'.join(buf).strip() + '\n')
                i = j + 1
                continue

        # headings
        m_h1 = re.match(r'^\s*##\s+(.*)$', line)
        m_h2 = re.match(r'^\s*###\s+(.*)$', line)
        if m_h1:
            # close itemize if open
            if in_itemize:
                out.append('\\end{itemize}')
                in_itemize = False
            content = transform_text_preserving_math(m_h1.group(1).strip())
            out.append(f'\\section{{{content}}}')
            i += 1
            continue
        if m_h2:
            if in_itemize:
                out.append('\\end{itemize}')
                in_itemize = False
            content = transform_text_preserving_math(m_h2.group(1).strip())
            out.append(f'\\subsection{{{content}}}')
            i += 1
            continue

        # horizontal rule (---)
        if re.match(r'^\s*---\s*$', line):
            if in_itemize:
                out.append('\\end{itemize}')
                in_itemize = False
            out.append('\\hrule')
            i += 1
            continue

        # list item
        m_list = re.match(r'^\s*-\s+(.*)$', line)
        if m_list:
            if not in_itemize:
                out.append('\\begin{itemize}')
                in_itemize = True
            content = transform_text_preserving_math(m_list.group(1).strip())
            out.append('\\item ' + content)
            i += 1
            continue
        else:
            if in_itemize:
                out.append('\\end{itemize}')
                in_itemize = False

        # blank line -> paragraph break
        if line.strip() == '':
            out.append('')
            i += 1
            continue

        # detect "Imagen representativa: page_XXXX.png"
        m_img = re.match(r'^\s*Imagen representativa:\s*page[_\\]?(\d+)\.png', line)
        if m_img:
            page_num = m_img.group(1)
            imgpath = f"images/page_{page_num}.png"
            out.append(r"\begin{center}")
            out.append(rf"\includegraphics[width=1\textwidth]{{{imgpath}}}")
            out.append(r"\end{center}")
            i += 1
            continue

        # normal paragraph line: transform (handles inline math)
        transformed = transform_text_preserving_math(line)
        out.append(transformed)
        i += 1

    # close open itemize
    if in_itemize:
        out.append('\\end{itemize}')
        in_itemize = False

    return out

def md_to_latex_file(md_path: Path, out_path: Path, title: str):
    md_text = md_path.read_text(encoding='utf-8')
    lines = md_text.splitlines()
    tex_lines = [PREAMBLE.replace('__TITLE__', title)]
    converted = convert_md_lines(lines)
    tex_lines.extend(converted)
    tex_lines.append(ENDDOC)
    out_path.write_text('\n'.join(tex_lines), encoding='utf-8')
    print(f"Wrote {out_path}")

def main():
    p = argparse.ArgumentParser(description='Simple Markdown->LaTeX for exam solutions')
    p.add_argument('--md', required=True, help='input markdown file')
    p.add_argument('--out', required=True, help='output .tex file')
    p.add_argument('--title', default='Soluciones', help='document title')
    args = p.parse_args()

    md_path = Path(args.md)
    out_path = Path(args.out)
    if not md_path.exists():
        print("Input file does not exist:", md_path)
        return
    md_to_latex_file(md_path, out_path, args.title)

if __name__ == '__main__':
    main()
