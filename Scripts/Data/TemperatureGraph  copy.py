"""
Este script grafica los datos de temperatura de la CPU de la Raspberry Pi 4, 
la temperatura de la VPU de OAK-D y la temperatura de su disipador térmico.

Una estructura de los datos es la siguiente:

Tiempo (s),VPU (ºC),CPU (ºC),MAX6675 (ºC)
t1,vpu1,cpu1,max6675_1
t2,vpu2,cpu2,max6675_2
t3,vpu3,cpu3,max6675_3
...
tn,vpun,cpun,max6675_n

La otra estructura de los datos es la siguiente:

times,z,d,h,v,nearest_labels,haptic_messages,buzzer_messages,chipTemperature,max6675Temperature,cpuTemperature
t1,z1,d1,h1,v1,nearest_labels1,haptic_messages1,buzzer_messages1,chipTemperature1,max6675Temperature1,cpuTemperature1
t2,z2,d2,h2,v2,nearest_labels2,haptic_messages2,buzzer_messages2,chipTemperature2,max6675Temperature2,cpuTemperature2
t3,z3,d3,h3,v3,nearest_labels3,haptic_messages3,buzzer_messages3,chipTemperature3,max6675Temperature3,cpuTemperature3
...
tn,zn,dn,hn,vn,nearest_labelsn,haptic_messagesn,buzzer_messagesn,chipTemperaturen,max6675Temperaturen,cpuTemperaturen

La idea es graficar unicamente los datod de temperaturas y sus respectivos ajustes de curva 
"""


import pandas as pd
import os
from matplotlib import rc
import matplotlib.pylab as plt
import numpy as np

# Set the font to Computer Modern 12pt
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 26})
rc('text', usetex=True)
rc('legend', fontsize=20)
plt.rc('text.latex', preamble=r'\usepackage{wasysym}')

# Ejecutar el script en la ruta actual
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Leer los datos del archivos .csv
data = pd.read_csv("Temperaturas2.csv")

# Graficar distancia horizontal y vertical del centro de la imagen a la detección más cercana
fig = plt.figure(figsize=(16, 8))
# Ajustar el tamaño de la grafica al tamaño de la figura fig
deltaX, deltaY = 0.065, .09
offsetX, offsetY = 0.05, 0.03
ax = fig.add_axes([deltaX, deltaY + offsetY, 1 - 2 * deltaX, 1 - 2 * deltaY])
plt.grid(color='gray', linestyle=':', alpha=.2) # Agregar grid minor con lineas, punteadas y color gris claro


# define my model fitting function
def func(x, T_0, T_inf, tau): # función
    return (T_inf - T_0) * (1 - np.exp(-x/tau)) + T_0

# define my data
start_sample = 50
samples = len(data) # número de muestras
n = np.linspace(0, 3600, samples - start_sample) # tiempo en segundos
times = data.times[start_sample:samples] # vector de tiempos	
chipTemperature = data.chipTemperature[start_sample:samples] # vector de temperaturas
max6675Temperature = data.max6675Temperature[start_sample:samples] # vector de temperaturas
cpuTemperature = data.cpuTemperature[start_sample:samples] # vector de temperaturas

# define my initial guess for the parameters
p0_chipTemperature = [37, 57, 500]
p0_max6675Temperature = [20, 27, 500]
p0_cpuTemperature = [59.5, 66.6604982, 450]

# fit the data to the model
from scipy.optimize import curve_fit
# fit the data to the model for chipTemperature
popt_chipTemperature, pcov_chipTemperature = curve_fit(func, n, chipTemperature, p0_chipTemperature)
# fit the data to the model for max6675Temperature
popt_max6675Temperature, pcov_max6675Temperature = curve_fit(func, n, max6675Temperature, p0_max6675Temperature)
# fit the data to the model for cpuTemperature
popt_cpuTemperature, pcov_cpuTemperature = curve_fit(func, n, cpuTemperature, p0_cpuTemperature)

print("popt_chipTemperature: ", popt_chipTemperature)
print("popt_max6675Temperature: ", popt_max6675Temperature)
print("popt_cpuTemperature: ", popt_cpuTemperature)

# plot the data
ax.plot(n, cpuTemperature, 'ro', markersize=2, label=r'CPU Temperature')
ax.plot(n, chipTemperature, 'c.', markersize=2, label=r"OAK-D Chip Temperature")
ax.plot(n, max6675Temperature, 'g.', markersize=2, label=r'Thermocouple Temperature')

# plot the fitted model
ax.plot(n, func(n, *popt_chipTemperature), 'k-', label="_nolegend_")
ax.plot(n, func(n, *popt_max6675Temperature), 'k-', label="_nolegend_")
#ax.plot(n, func(n, *popt_cpuTemperature), 'k-', label="_nolegend_")

# Asintotas horizontales en el valor de T_inf para cada curva
ax.axhline(y=popt_chipTemperature[1], color='k', alpha=0.5, linestyle='--', linewidth=1, label="_nolegend_")
ax.axhline(y=popt_max6675Temperature[1], color='k', alpha=0.5, linestyle='--', linewidth=1, label="_nolegend_")
ax.axhline(y=popt_cpuTemperature[1], color='k', alpha=0.5, linestyle='--', linewidth=1, label="_nolegend_")

# colodar los valores de T_inf con 2 desimales en el ylabel de la derecha
ax.text(1, 0.95, str(round(popt_cpuTemperature[1], 2)) + r'$^{\circ}$C', horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, fontsize=15)
ax.text(1, 0.77, str(round(popt_chipTemperature[1], 2)) + r'$^{\circ}$C', horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, fontsize=15)	
ax.text(1, 0.25, str(round(popt_max6675Temperature[1], 2)) + r'$^{\circ}$C', horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, fontsize=15)



plt.title(r"Operating system temperature measurements")
plt.xlabel(r'Time [s]')
plt.ylabel(r'Temperature [°C]')
plt.xlim(0, 3800)
plt.legend(loc='best')
plt.savefig('Tempetature2.svg', format='svg', dpi=1200)
plt.savefig('Tempetature2.png', format='png', dpi=1200)
plt.savefig('Tempetature2.pdf', format='pdf', dpi=1200)
plt.show()