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

# Leer los datos de los archivos .csv
data = pd.read_csv("/home/user/Trabajo de titulación/BVI-TEST/Scripts/Data/Temperatures_HandsGPU.csv")

# Graficar distancia horizontal y vertical del centro de la imagen a la detección más cercana
fig = plt.figure(figsize=(16, 8))
# Ajustar el tamaño de la grafica al tamaño de la figura figp0_MAX6675_TEMPERATURE
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
times = data.TIMES[start_sample:] # tiempo en segundos
VPU_TEMPERATURE = data.VPU[start_sample:] # temperatura del chip en °C


# define my initial guess for the parameters
p0_VPU_TEMPERATURE = [37, 57, 500]
p0_MAX6675_TEMPERATURE = [20, 27, 500]
p0_CPU_TEMPERATURE = [59.5, 66.6604982, 450]

# fit the data to the model
from scipy.optimize import curve_fit
# fit the data to the model for VPU_TEMPERATURE
popt_VPU_TEMPERATURE, pcov_VPU_TEMPERATURE = curve_fit(func, n, VPU_TEMPERATURE, p0_VPU_TEMPERATURE)
# fit the data to the model for MAX6675_TEMPERATURE
popt_MAX6675_TEMPERATURE, pcov_MAX6675_TEMPERATURE = curve_fit(func, n, MAX6675_TEMPERATURE, p0_MAX6675_TEMPERATURE)
# fit the data to the model for CPU_TEMPERATURE
popt_CPU_TEMPERATURE, pcov_CPU_TEMPERATURE = curve_fit(func, n, CPU_TEMPERATURE, p0_CPU_TEMPERATURE)

print("popt_VPU_TEMPERATURE: ", popt_VPU_TEMPERATURE, "pcov_VPU_TEMPERATURE: ", pcov_VPU_TEMPERATURE)
print("popt_MAX6675_TEMPERATURE: ", popt_MAX6675_TEMPERATURE, "pcov_MAX6675_TEMPERATURE: ", pcov_MAX6675_TEMPERATURE)
print("popt_CPU_TEMPERATURE: ", popt_CPU_TEMPERATURE, "pcov_CPU_TEMPERATURE: ", pcov_CPU_TEMPERATURE)

# plot the data
ax.plot(n, CPU_TEMPERATURE, 'r.', markersize=2, label=r'CPU Temperature')
ax.plot(n, VPU_TEMPERATURE, 'c.', markersize=2, label=r"OAK-D Chip Temperature")
ax.plot(n, MAX6675_TEMPERATURE, 'g.', markersize=2, label=r'Thermocouple Temperature')

# plot the fitted model
ax.plot(n, func(n, *popt_VPU_TEMPERATURE), 'k-', label="_nolegend_")
ax.plot(n, func(n, *popt_MAX6675_TEMPERATURE), 'k-', label="_nolegend_")
#ax.plot(n, func(n, *popt_CPU_TEMPERATURE), 'k-', label="_nolegend_")

# Asintotas horizontales en el valor de T_inf para cada curva
ax.axhline(y=popt_VPU_TEMPERATURE[1], color='k', alpha=0.5, linestyle='--', linewidth=1, label="_nolegend_")
ax.axhline(y=popt_MAX6675_TEMPERATURE[1], color='k', alpha=0.5, linestyle='--', linewidth=1, label="_nolegend_")
ax.axhline(y=popt_CPU_TEMPERATURE[1], color='k', alpha=0.5, linestyle='--', linewidth=1, label="_nolegend_")

# colodar los valores de T_inf con 2 desimales en el ylabel de la derecha
ax.text(1, 0.95, str(round(popt_CPU_TEMPERATURE[1], 2)) + r'$^{\circ}$C', horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, fontsize=15)
ax.text(1, 0.77, str(round(popt_VPU_TEMPERATURE[1], 2)) + r'$^{\circ}$C', horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, fontsize=15)	
ax.text(1, 0.25, str(round(popt_MAX6675_TEMPERATURE[1], 2)) + r'$^{\circ}$C', horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, fontsize=15)


plt.title(r"Operating system temperature measurements")
plt.xlabel(r'Time [s]')
plt.ylabel(r'Temperature [°C]')
plt.xlim(0, 3800)
plt.legend(loc='best')
plt.savefig('Tempetature2.svg', format='svg', dpi=1200)
plt.savefig('Tempetature2.png', format='png', dpi=1200)
plt.savefig('Tempetature2.pdf', format='pdf', dpi=1200)
plt.show()