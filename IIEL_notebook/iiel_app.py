import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2*R*asin(sqrt(a))

def generar_datos():
    n = 10
    df = pd.DataFrame({
        'municipio': [f'M{i}' for i in range(n)],
        'lat': np.random.uniform(35, 45, n),
        'lon': np.random.uniform(-10, 3, n),
        'salario': np.random.randint(15000, 40000, n),
        'poblacion': np.random.randint(1000, 50000, n)
    })
    df.to_csv('datos_sinteticos.csv', index=False)
    messagebox.showinfo('Generado', 'Datos sintéticos guardados como datos_sinteticos.csv')

def abrir_archivo():
    path = filedialog.askopenfilename(filetypes=[('CSV Files', '*.csv')])
    if path:
        df = pd.read_csv(path)
        messagebox.showinfo('Cargado', f'{len(df)} filas cargadas')
        print(df.head())

def ejecutar_calculo():
    messagebox.showinfo('Ejecutar', 'Aquí se ejecutaría el cálculo del índice IIEL.')

root = tk.Tk()
root.title('IIEL Toolkit GUI')

tk.Button(root, text='Generar datos sintéticos', command=generar_datos).pack(pady=5)
tk.Button(root, text='Cargar CSV', command=abrir_archivo).pack(pady=5)
tk.Button(root, text='Ejecutar cálculo', command=ejecutar_calculo).pack(pady=5)
tk.Button(root, text='Salir', command=root.quit).pack(pady=10)

root.mainloop()
