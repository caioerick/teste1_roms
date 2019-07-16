#!/usr/bin/env python
# coding: utf-8

# In[1]:

# rotina para checar a batimetria que estou usando 
# no meu experimento climatologico, bruta e apos filtragem

import scipy.io as sp
import matplotlib.pyplot as plt
import numpy as np
import sys
#from cookb_signalsmooth import smooth
import netCDF4 as nc


##############################################
# edit these options:                        #
exptname = 'FM1'                             #
case = 'teste1'                              #
##############################################

print('\n')
print('Experimento:', exptname)
print('Case:', case)
print('\n')

# arquivo de batimetria original
aux = sp.loadmat('../essentials/etopo1_teste1.mat')

topo = aux['topo'][:]
lon = aux['lon'][:]
lat = aux['lat'][:]

# latitude para fazer o plot
lat_sec = -4

lon,lat = np.meshgrid(lon,lat)


# plotting maps

lv = np.arange(-6000,1,100)

plt.figure(figsize=(7,4))
plt.title('Batimetria da Área de Estudo')
cs = plt.contourf(lon, lat, topo, levels=lv)
plt.ylim(-11,-1)
plt.xlim(-40,-26)
plt.colorbar()
#plt.show()
plt.savefig('../figures/bathymetry/batimetria_etopo1.png', dpi=140)      
plt.close()


# plotting profiles

print('Iniciando o plot do perfil de batimetria...')

lat = lat[:,0]
lat.shape = (lat.size,1)
lon = lon[0,:]
lon.shape = (lon.size,1)

f = np.where(lat>=lat_sec)
f = f[0]
f = f.min()
plt.figure(),
plt.plot(lon,topo[f,:])
plt.grid('on')
plt.title('lat = '+ str(np.round(lat[f],2)))
savebat = plt.savefig(('../figures/bathymetry/batim_etopo1_perfil_lat' + str(np.round(lat[f],2)) + '.png'),dpi = 140)

if(savebat):
	print('Figura salva em o plot do perfil de batimetria...\n\n\n\n\n')

plt.close()
del f
