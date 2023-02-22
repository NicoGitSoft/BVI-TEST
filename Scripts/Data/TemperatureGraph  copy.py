"""
Script para graficar los datos dos archivos llamados:

file1: "system temperatures without distributed processing.csv"
file2: "system temperatures with distributed processing.csv"

los cuales contienen los datos de temperatura de la VPU del sensor OAK-D, 
la temperatura de la CPU de la Raspberry Pi 4 y la temperatura del sensor MAX6675.

Los datos se encuentran en archivos .csv con el siguiente formato:

TIMES,VPU,CPU,MAX6675
t1,vpu1,cpu1,max6675_1
t2,vpu2,cpu2,max6675_2
...
tn,vpun,cpun,max6675_n

Las graficas usan formato LaTeX para los ejes y leyendas, por otro lado,
se grafican dos subplots para comparar los datos de las temperaturas de lo los dos archivos
"""

import pandas as pd
from matplotlib import rc
import matplotlib.pylab as plt
import numpy as np
from scipy.optimize import curve_fit
import os

# Set the font to Computer Modern 12pt
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 26})
rc('text', usetex=True)
rc('legend', fontsize=20)
plt.rc('text.latex', preamble=r'\usepackage{wasysym}')

# Ejecutar el script en la ruta actual donde se encuentran los datos
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Leer los datos
file1 = pd.read_csv('system temperatures without distributed processing.csv')
file2 = pd.read_csv('system temperatures with distributed processing.csv')

# Obtener los datos de las temperaturas
start_sample = 50

# datos del archivo 1 
times_1 = file1['TIMES'][start_sample:]
vpu_1 = file1['VPU'][start_sample:]
cpu_1 = file1['CPU'][start_sample:]
max6675_1 = file1['THERMOCOUPLE'][start_sample:]

# datos del archivo 2
times_2 = file2['TIMES'][start_sample:]
vpu_2 = file2['VPU'][start_sample:]
cpu_2 = file2['CPU'][start_sample:]
max6675_2 = file2['THERMOCOUPLE'][start_sample:]

# Función para ajustes de curva exponencial
def func(x, T_inf, T_0, tau):
    return (T_inf - T_0) * (1 - np.exp(-x/tau)) + T_0


p0_chipTemperature = [37, 57, 500]
p0_max6675Temperature = [20, 27, 500]
p0_cpuTemperature = [59.5, 66.6604982, 450]

# Ajuste de curva exponencial para los datos del archivo 1
popt_vpu_1, pcov_vpu_1 = curve_fit(func, times_1, vpu_1, p0=[37, 57, 500])
popt_cpu_1, pcov_cpu_1 = curve_fit(func, times_1, cpu_1, p0=[20, 27, 500])
popt_max6675_1, pcov_max6675_1 = curve_fit(func, times_1, max6675_1, p0=[59.5, 66.6604982, 450])

# Ajuste de curva exponencial para los datos del archivo 2
popt_vpu_2, pcov_vpu_2 = curve_fit(func, times_2, vpu_2, p0=[37, 57, 500])
popt_cpu_2, pcov_cpu_2 = curve_fit(func, times_2, cpu_2, p0=[20, 27, 500])
popt_max6675_2, pcov_max6675_2 = curve_fit(func, times_2, max6675_2, p0=[59.5, 66.6604982, 450])

# Mostrar por consola los parámetros de ajuste pcov
print('Archivo 1')
print('VPU: ', popt_vpu_1)
print('CPU: ', popt_cpu_1)
print('MAX6675: ', popt_max6675_1)
print('Archivo 2')
print('VPU: ', popt_vpu_2)
print('CPU: ', popt_cpu_2)
print('MAX6675: ', popt_max6675_2)

# Crear la figura
fig, axs = plt.subplots(3, 1, figsize=(20, 20), sharex=True)

# Graficar los datos de la VPU y sus ajustes de curva exponencial
#axs[0].plot(times_1, vpu_1, 'r.', markersize=2, label=r'VPU Temperature')
axs[0].plot(times_1, func(times_1, *popt_vpu_1), 'r-', label=r'VPU Fit')
#axs[0].plot(times_2, vpu_2, 'b.', markersize=2, label=r'VPU Temperature')
axs[0].plot(times_2, func(times_2, *popt_vpu_2), 'b-', label=r'VPU Fit')
axs[0].set_ylabel(r'VPU Temperature ($^\circ$C)')
axs[0].legend()

# Graficar los datos de la CPU y sus ajustes de curva exponencial
#axs[1].plot(times_1, cpu_1, 'r.', markersize=2, label=r'CPU Temperature')
axs[1].plot(times_1, func(times_1, *popt_cpu_1), 'r-', label=r'CPU Fit')
#axs[1].plot(times_2, cpu_2, 'b.', markersize=2, label=r'CPU Temperature')
axs[1].plot(times_2, func(times_2, *popt_cpu_2), 'b-', label=r'CPU Fit')
axs[1].set_ylabel(r'CPU Temperature ($^\circ$C)')
axs[1].legend()

# Graficar los datos del MAX6675 y sus ajustes de curva exponencial
#axs[2].plot(times_1, max6675_1, 'r.', markersize=2, label=r'MAX6675 Temperature')
axs[2].plot(times_1, func(times_1, *popt_max6675_1), 'r-', label=r'MAX6675 Fit')
#axs[2].plot(times_2, max6675_2, 'b.', markersize=2, label=r'MAX6675 Temperature')
axs[2].plot(times_2, func(times_2, *popt_max6675_2), 'b-', label=r'MAX6675 Fit')
axs[2].set_ylabel(r'MAX6675 Temperature ($^\circ$C)')
axs[2].set_xlabel(r'Time (s)')
axs[2].legend()

# Mostrar la figura
plt.show()