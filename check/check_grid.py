#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# rotina para checar a batimetria que estou usando 
# no meu experimento climatologico, bruta e apos filtragem

from roms_setup import run_setup
import scipy.io as sp
import matplotlib.pyplot as plt
import numpy as np
import sys
#from cookb_signalsmooth import smooth
import netCDF4 as nc


##############################################
# edit these options:                        #
exptname = 'tcc'                             #
case = 'teste1'                              #
##############################################

print('\n')
print('Experimento:', exptname)
print('Case:', case)
print('\n')

print('--> Lendo o arquivo de configuração...\n')
run = run_setup('../config.setup')

# arquivo de batimetria original
aux = sp.loadmat(run.topo_filename)

topo = aux['topo'][:]
lon = aux['lon'][:]
lat = aux['lat'][:]

# latitude para fazer o plot
lat_sec = -4

lon,lat = np.meshgrid(lon,lat)

# plotting maps
print('--> Gerando o mapa com a batimetria...\n')
lv = np.arange(-6000,1,100)

plt.figure(figsize=(7,4))
plt.title('Batimetria da Área de Estudo')
cs = plt.pcolormesh(lon, lat, topo)
plt.ylim(-11,-1)
plt.xlim(-40,-26)
plt.colorbar()
plt.savefig('../figures/bathymetry/batimetria_etopo1.png', dpi=140)      
plt.close()

print('--> Mapa de batimetria salvo com sucesso!\n')

plt.close()





# plotting profiles
print('--> Iniciando o plot do perfil de batimetria...\n')

"""
lat = lat[:,0]
lat.shape = (lat.size,1)
lon = lon[0,:]
lon.shape = (lon.size,1)
"""

lat = np.array(lat[:,0])
lon= np.array(lon[0,:])

f = np.where(lat>=lat_sec)
f = f[0]
f = f.min()
plt.figure(),
plt.plot(lon,topo[f,:])
plt.grid()
plt.title('Latitude: '+ str(np.round(lat[f],2)))

plt.savefig(('../figures/bathymetry/batim_etopo1_perfil_lat' + str(np.round(lat[f],2)) + '.png'),dpi = 140)

print('--> Perfil de batimetria em {} salvo com sucesso!\n\n'.format(lat[f],2))

plt.close()
del f
