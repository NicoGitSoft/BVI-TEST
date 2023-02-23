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

# Ejecutar el script en la ruta actual donde se encuentran los datos
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Leer los datos
file1 = pd.read_csv('Temperatures_HandsGPU.csv')
file2 = pd.read_csv('system temperatures without distributed processing.csv')

# Obtener los datos de las temperaturas
start_sample = 50

# vertor de nuestras
n1 = np.arange(0, len(file1['TIMES'][start_sample:]))
n2 = np.arange(0, len(file2['TIMES'][start_sample:]))

# datos del archivo 1 
VPU_TEMPERATURE_1 = file1['VPU'][start_sample:]
CPU_TEMPERATURE_1 = file1['CPU'][start_sample:]
MAX6675_TEMPERATURE_1 = file1['THERMOCOUPLE'][start_sample:]

# datos del archivo 2
VPU_TEMPERATURE_2 = file2['VPU'][start_sample:]
CPU_TEMPERATURE_2 = file2['CPU'][start_sample:]
MAX6675_TEMPERATURE_2 = file2['THERMOCOUPLE'][start_sample:]

# Función para ajustes de curva exponencial
def func(x, T_0, T_inf, tau):
    return (T_inf - T_0) * (1 - np.exp(-x/tau)) + T_0

# Puntos iniciales para ajuste de curva exponencial
p0_VPU_TEMPERATURE_1 = p0_VPU_TEMPERATURE_2 = [37, 57, 500]
p0_CPU_TEMPERATURE_1 = p0_CPU_TEMPERATURE_2 = [59.5, 66.6604982, 450]
p0_MAX6675_TEMPERATURE_1 = p0_MAX6675_TEMPERATURE_2 = [20, 27, 500]

# Ajuste de curva exponencial para los datos del archivo 1
popt_VPU_TEMPERATURE_1, pcov_VPU_TEMPERATURE_1 = curve_fit(func, n1, VPU_TEMPERATURE_1, p0=p0_VPU_TEMPERATURE_1)
popt_CPU_TEMPERATURE_1, pcov_CPU_TEMPERATURE_1 = curve_fit(func, n1, CPU_TEMPERATURE_1, p0=p0_CPU_TEMPERATURE_1)
popt_MAX6675_TEMPERATURE_1, pcov_MAX6675_TEMPERATURE_1 = curve_fit(func, n1, MAX6675_TEMPERATURE_1, p0=p0_MAX6675_TEMPERATURE_1)

# Ajuste de curva exponencial para los datos del archivo 2
popt_VPU_TEMPERATURE_2, pcov_VPU_TEMPERATURE_2 = curve_fit(func, n2, VPU_TEMPERATURE_2, p0=p0_VPU_TEMPERATURE_2)
popt_CPU_TEMPERATURE_2, pcov_CPU_TEMPERATURE_2 = curve_fit(func, n2, CPU_TEMPERATURE_2, p0=p0_CPU_TEMPERATURE_2)
popt_MAX6675_TEMPERATURE_2, pcov_MAX6675_TEMPERATURE_2 = curve_fit(func, n2, MAX6675_TEMPERATURE_2, p0=p0_MAX6675_TEMPERATURE_2)

# Mostrar por consola los parámetros de ajuste pcov
print('Archivo 1\n VPU: ', popt_VPU_TEMPERATURE_1, '\n CPU: ', popt_CPU_TEMPERATURE_1, '\n MAX6675: ', popt_MAX6675_TEMPERATURE_1)
print('Archivo 2\n VPU: ', popt_VPU_TEMPERATURE_2, '\n CPU: ', popt_CPU_TEMPERATURE_2, '\n MAX6675: ', popt_MAX6675_TEMPERATURE_2)


########################## GRAFICAS ##########################

# Configuración de las graficas
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 26})
rc('text', usetex=True)
rc('legend', fontsize=20)
plt.rc('text.latex', preamble=r'\usepackage{wasysym}')

# Crear una figura
figure_1 = plt.figure(figsize=(16, 8))

# Ajustar el tamaño de la grafica al tamaño de la figura
deltaX, deltaY = 0.065, .09
offsetX, offsetY = 0.05, 0.03
ax = figure_1.add_axes([deltaX, deltaY + offsetY, 1 - 2 * deltaX, 1 - 2 * deltaY])

# Agrergar grid principal con lineas, punteadas y color gris claro
plt.grid(color='gray', linestyle=':', alpha=.2) 

# Graficar los datos de temperatura del archivo 1
ax.plot(n1, CPU_TEMPERATURE_1, 'r.', markersize=2, label=r'CPU Temperature')
ax.plot(n1, VPU_TEMPERATURE_1, 'c.', markersize=2, label=r"OAK-D Chip Temperature")
ax.plot(n1, MAX6675_TEMPERATURE_1, 'g.', markersize=2, label=r'Thermocouple Temperature')

# Graficar los ajustes de curva exponencial para los datos del archivo 1
ax.plot(n1, func(n1, *popt_VPU_TEMPERATURE_1), 'k-', label="_nolegend_")
ax.plot(n1, func(n1, *popt_MAX6675_TEMPERATURE_1), 'k-', label="_nolegend_")
ax.plot(n1, func(n1, *popt_CPU_TEMPERATURE_1), 'k-', label="_nolegend_")

# Asintotas horizontales en el valor de T_inf para cada curva
ax.axhline(y=popt_VPU_TEMPERATURE_1[1], color='k', alpha=0.5, linestyle='--', linewidth=1, label="_nolegend_")
ax.axhline(y=popt_MAX6675_TEMPERATURE_1[1], color='k', alpha=0.5, linestyle='--', linewidth=1, label="_nolegend_")
ax.axhline(y=popt_CPU_TEMPERATURE_1[1], color='k', alpha=0.5, linestyle='--', linewidth=1, label="_nolegend_")

# Definir limites de ejes 
y_min = min([popt_CPU_TEMPERATURE_1[0], popt_VPU_TEMPERATURE_1[0], popt_MAX6675_TEMPERATURE_1[0]])
y_max = max([popt_CPU_TEMPERATURE_1[1], popt_VPU_TEMPERATURE_1[1], popt_MAX6675_TEMPERATURE_1[1]])
y_delta = (y_max - y_min) * 0.1
ax.set_ylim(y_min - y_delta, y_max + y_delta)
plt.xlim(0, 3800)

# colodar los valores de T_inf con 2 desimales en el ylabel de la derecha

yText_CPU_TEMPERATURE_1 = (popt_CPU_TEMPERATURE_1[1]-y_min+y_delta)/(y_max-y_min+2*y_delta)
yText_VPU_TEMPERATURE_1 = (popt_VPU_TEMPERATURE_1[1]-y_min+y_delta)/(y_max-y_min+2*y_delta)
yText_MAX6675_TEMPERATURE_1 = (popt_MAX6675_TEMPERATURE_1[1]-y_min+y_delta)/(y_max-y_min+2*y_delta)

ax.text(1, yText_CPU_TEMPERATURE_1, str(round(popt_CPU_TEMPERATURE_1[1], 2)) + r'$^{\circ}$C', horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes, fontsize=15)
ax.text(1, yText_VPU_TEMPERATURE_1, str(round(popt_VPU_TEMPERATURE_1[1], 2)) + r'$^{\circ}$C', horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes, fontsize=15)	
ax.text(1, yText_MAX6675_TEMPERATURE_1, str(round(popt_MAX6675_TEMPERATURE_1[1], 2)) + r'$^{\circ}$C', horizontalalignment='right', verticalalignment='bottom', transform=ax.transAxes, fontsize=15)

# Mostrar el titulo, etiquetas de los ejes y leyenda
plt.title(r"System temperatures with distributed processing")
plt.xlabel(r'Time [s]')
plt.ylabel(r'Temperature [°C]')

plt.legend(loc='best')
#plt.savefig('Tempetature.svg', format='svg', dpi=1200)
plt.savefig('Temperatures_HandsGPU.png', format='png', dpi=1200)
#plt.savefig('Tempetature.pdf', format='pdf', dpi=1200)
plt.show()