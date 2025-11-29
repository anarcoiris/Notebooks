
# IIEL - Interactive Index of Local Economic Impact

This repository contains a Jupyter notebook and helper files to compute and analyze the
local economic impact of large energy infrastructures (IIEL).

## Files
- `notebook_IIEL.ipynb` - The main notebook. Run sequentially in JupyterLab or Jupyter Notebook.
- `requirements.txt` - Python dependencies.
- `README.md` - This file.

## Quick start
1. Create a virtual environment and install dependencies:
```
python -m venv venv
source venv/bin/activate       # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Launch Jupyter:
```
jupyter lab
```

3. Open `notebook_IIEL.ipynb` and run cells in order. The notebook:
- Generates synthetic data (if you don't supply CSVs).
- Computes distance cache.
- Builds kernel-weighted treatment intensity.
- Runs FE panel and event-study (simplified).
- Provides a Tkinter GUI (only when running the script directly).

Notes:
- The notebook is educational and contains placeholders where you may integrate more advanced methods (2SLS, SAR with pysal, Conley SEs).
- Tkinter GUIs usually run better when executed as a Python script in a local environment, not inside hosted Jupyter services.

