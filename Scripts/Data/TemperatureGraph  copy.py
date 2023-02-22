"""
Script para graficar los datos de temperatura de la VPU del sensor OAK-D, 
la temperatura de la CPU de la Raspberry Pi 4 y la temperatura del sensor MAX6675.

Los datos se encuentran en archivos .csv con el siguiente formato:

TIMES,VPU,CPU,MAX6675
t1,vpu1,cpu1,max6675_1
t2,vpu2,cpu2,max6675_2
...
tn,vpun,cpun,max6675_n

Las graficas usan formato LaTeX para los ejes y leyendas.
"""

import pandas as pd
import os
from matplotlib import rc
import matplotlib.pylab as plt
import numpy as np


# Función para el ajuste de curva
def func(x, T_0, T_inf, tau): # función
    return (T_inf - T_0) * (1 - np.exp(-x/tau)) + T_0


# Set the font to Computer Modern 12pt
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 26})
rc('text', usetex=True)
rc('legend', fontsize=20)
plt.rc('text.latex', preamble=r'\usepackage{wasysym}')

# Ejecutar el script en la ruta actual donde se encuentran los datos
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Leer los datos
data1 = pd.read_csv('Temperaturas


# Recortar los datos desde la muestra 50
start_sample = 50