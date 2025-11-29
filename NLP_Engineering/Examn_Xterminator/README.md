# Examn Xterminator

## Requirements:
pip install pymupdf Pillow pytesseract requests rapidfuzz

setx OPENAI_API_KEY (your api key)

## How to use:
python pdf_exam.py --pdf /route/to/your_merged_examns.pdf --out ./"folder name" --sleep (stream delay in seconds, default 1, 5 works fine)

python autolatex.py --md "./folder/solutions.md" --out your_output.tex --title "Exámenes de... (whatever)"


## To-Do Ideas:
Adding index (and curate the section, subsection, etc...)

Adding resume

Analize by examn structure (instead of pages)

Checking for multiple languages

PDF compiler + Unified scripts (im using overleaf for now)

Embedding in a LaTeX template

...

GUI, TTS integration...

# Examn_Xterminator

## Resumen
Examn_Xterminator es una herramienta que permite analizar exámenes en formato PDF de años anteriores. Utiliza la API de OpenAI para identificar la frecuencia de problemas similares, resolverlos e indexarlos, generando un documento en LaTeX con las soluciones ordenadas por frecuencia.

## Requisitos
- Python 3.6 o superior
- Bibliotecas necesarias:
  - `requests`
  - `PyPDF2`
  - `latex`
- Acceso a la API de OpenAI (clave de API requerida)

## Instalación
1. Clona el repositorio:
   ```bash
      git clone https://github.com/tu_usuario/Examn_Xterminator.git
   ```
2. Navega al directorio del proyecto:
   ```bash
   cd Examn_Xterminator
   ```
3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Ejecución
Para ejecutar la herramienta, utiliza el siguiente comando:
```bash
python main.py <ruta_al_pdf>
```
Reemplaza `<ruta_al_pdf>` con la ruta del archivo PDF que deseas analizar.

## Ejemplo de uso
```bash
python main.py /ruta/a/tu/examen.pdf
```
Esto generará un documento LaTeX con las soluciones a los problemas encontrados en el examen.

## Estructura del proyecto
```
Examn_Xterminator/
│
├── AuToLaTeX.py
│   - Funciones: 8
│   - Docstring de módulo: sí
│
├── AuToTeX_UNED.py
│   - Funciones: 8
│   - Docstring de módulo: sí
│
├── pdf_exam.py
│   - Funciones: 10
│   - Docstring de módulo: sí
│└── pdf_merger.py
    - Funciones: 6
    - Docstring de módulo: sí
```

## Desarrollo/Contribución
Si deseas contribuir al proyecto, sigue estos pasos:
1. Haz un fork del repositorio.
2. Crea una nueva rama:
   ```bash
   git checkout -b nombre_de_tu_rama
   ```
3. Realiza tus cambios y haz commit:
   ```bash
   git commit -m "Descripción de los cambios"
   ```
4. Envía un pull request.

## Licencia
Este proyecto está bajo la Licencia MIT. Para más detalles, consulta el archivo `LICENSE`.

## Resultados
Al ejecutar la herramienta, se generará un documento en LaTeX que contiene:
- Un índice de problemas analizados.
- Soluciones a los problemas, ordenadas por frecuencia.
- Un formato limpio y profesional listo para ser utilizado.


