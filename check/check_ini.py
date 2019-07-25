#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# script to check ROMS initial fields

import scipy.io as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys
#from cookb_signalsmooth import smooth
import netCDF4 as nc
from roms_setup import run_setup, get_depths, zlevs, near
from matplotlib.mlab import griddata
from mpl_toolkits.basemap import Basemap
import math

# Adicionado por Caio
from scipy.interpolate import griddata
plt.close('all')

# tentar pegar direto do config.setup depois
#############################################################
# edit these options:             '                                     #
exptname = 'tcc_teste1'
l = 0    # time-step (0 for initial fields)                 #
lev = [-1,-200,-500,-1000,-4000] # levels for plotting      #
#############################################################

print('--> Lendo o arquivo de configuração...\n')
#run = run_setup('' + exptname + '.setup')
run = run_setup('../config.setup')

print('\n') 
print('Experiment ' + exptname)
print('\n') 

# plots parameters
lev = np.array(lev)

d  = 4  # subsampling for quiver plots
sc = .5 # scale for quiver plots

ni          = np.array([-9.9,-1])    # latitude, in case PLOT = 2
nj          = np.array([-39,-27])  # longitude, in case PLOT = 3 
dz          = 10          # delta z [m] for S -- > Z vertical interpolation
Zlim        = (-1500, 0)  # z axis limits
vsc         = (0, 31)
vst         = 2
levelsp     = np.arange(0.1,1,0.1)
levelsn     = np.arange(-1,-0.1,0.1)

print('Loading data ...\n')

grdfile  = nc.Dataset('../input/' + exptname + '_grd.nc')
lon   = grdfile.variables['lon_rho'][:]
lat   = grdfile.variables['lat_rho'][:]
latu  = grdfile.variables['lat_u'][:]
latv  = grdfile.variables['lat_v'][:]
lonu  = grdfile.variables['lon_u'][:]
lonv  = grdfile.variables['lon_v'][:]
h     = grdfile.variables['h'][:]

print('\n') 
print('Maximum depth of this run:' + str(h.max()) + 'm')
print('\n') 

inifile  = nc.Dataset('../input/' + exptname + '_ini.nc')
lon  = inifile.variables['lon_rho'][:]
lat  = inifile.variables['lat_rho'][:]
lonu = inifile.variables['lon_u'][:]
latu = inifile.variables['lat_u'][:]
lonv = inifile.variables['lon_v'][:]
latv = inifile.variables['lat_v'][:]
zeta = inifile.variables['zeta'][:]
ubar = inifile.variables['ubar'][:]
vbar = inifile.variables['vbar'][:]
U    = inifile.variables['u'][:]
V    = inifile.variables['v'][:]   
temp = inifile.variables['temp'][:]
salt = inifile.variables['salt'][:]

dx =lonv[0,1]-lonv[0,0] 

print('getting depths of s-levels...\n')

zt   = get_depths(inifile, grdfile, l, 'temp')
zu   = get_depths(inifile, grdfile, l, 'U')
zv   = get_depths(inifile, grdfile, l, 'V')

zeta = np.squeeze(zeta[l,...])
U    = np.squeeze(U[l,...])
V    = np.squeeze(V[l,...])
temp = np.squeeze(temp[l,...])
salt = np.squeeze(salt[l,...])
ubar = np.squeeze(ubar[l,...])
vbar = np.squeeze(vbar[l,...])
km, im, jm = temp.shape

# checking for NaN
f = np.where(np.isnan(U)==True); f = f[0]
if f:
    print('Warning!\n')
    print('Warning!\n')
    print('NaN found!!! The model grid might be bigger than the observation grid!!\n')
del f


m = Basemap(projection='merc',
		llcrnrlat = run.latmin, urcrnrlat = run.latmax, 
		llcrnrlon = run.lonmin, urcrnrlon = run.lonmax,
		 lat_ts=0, resolution='i')

lonm, latm = m(lon, lat)	
lonp = lonm[::d, ::d]
latp = latm[::d, ::d]

ax1,ay1 = m( run.lonmax -2, run.latmin + 0.5 )
tx1,ty1 = m( run.lonmin + 0.5, run.latmax - 4.6 )
tx2,ty2 = m( run.lonmin + 0.5, run.latmax - 5.0 )

print('run.lonmin ' + str(run.lonmin))

hscz = (-1, 1)  # zeta color scale
vsv = (-1, 1) # vertical slice velocity color scale
vst = 0.01       # interval for contourf
vsc = (0, 31)   # vertical slice temperature color scale# 0 31

for kk in range(0,lev.size,1):

	print('\n')
	print('level ' + str(kk+1) + '/' + str(lev.size) + ',   z = ' + str(lev[kk]))
	print('\n')

	if (lev[kk]*-1) > h.max():
		print(' \n')
		print('Sorry, can"t plot: this run just goes down to ' + str(h.max()) + ' meters')
		print(' \n')
		break

	# color limits
	if lev[kk] > -80:
		hsct = (24, 28) # temperature color scale 
		hscs = (34, 37) # salinity color scale 
	elif ((lev[kk] <= -80) & (lev[kk] > -200)):
		hsct = (10, 27) 
		hscs = (34, 37)
	elif ((lev[kk] <= -200) & (lev[kk] > -300)):
        	hsct = (11, 18)
        	hscs = (35.2, 36.4)
	elif ((lev[kk] <= -300) & (lev[kk] > -800)):
		hsct = (3, 15) 
		hscs = (34, 36) 
	else:
		hsct = (0, 6) 
		hscs = (34.2, 35)


	print('\n') 
	print('interpolating s -> z ... ')
	print('\n') 

	# setting the variables' dimensions
	u1 = 0*lonu; v1 = 0*lonv; t1 = 0*lon; s1 = 0*lon

	# interpolating vertically
	for a in range (0, im):
		for b in range(0, jm):
			t1[a,b] = np.interp(lev[kk], zt[:, a, b], temp[:, a, b] )
			s1[a,b] = np.interp(lev[kk], zt[:, a, b], salt[:, a, b] )

	for a in range (0, im):
		for b in range(0, jm-1):
			u1[a,b] = np.interp(lev[kk], zu[:, a, b], U[:, a, b] )

	for a in range (0, im-1):
		for b in range(0, jm):
			v1[a,b] = np.interp(lev[kk], zv[:, a, b], V[:, a, b] )

	# interpolating horizontally
	u1 = griddata((lonu.ravel(), latu.ravel()), u1.ravel(), (lon, lat)) 
	v1 = griddata((lonv.ravel(), latv.ravel()), v1.ravel(), (lon, lat))

	ubar1 = griddata((lonu.ravel(), latu.ravel()), ubar.ravel(), (lon, lat)) 
	vbar1 = griddata((lonv.ravel(), latv.ravel()), vbar.ravel(), (lon, lat))

	# masking in land values
	t1    = np.ma.masked_where(h < 1, t1)
	s1    = np.ma.masked_where(h < 1, s1)
	zeta  = np.ma.masked_where(zeta > 30, zeta)
	ubar1 = np.ma.masked_where(ubar1 > 1e30, ubar1)
	vbar1 = np.ma.masked_where(vbar1 > 1e30, vbar1)
	u1    = np.ma.masked_where(u1 > 1e30, u1)
	v1    = np.ma.masked_where(v1 > 1e30, v1)

	# subsampling
	u1   = u1[::d, ::d]
	v1   = v1[::d, ::d]
	ubar1   = ubar1[::d, ::d]
	vbar1   = vbar1[::d, ::d]

	# map temperature vs. velocity -----------------------------------------------
	prop = t1.copy()

	fig1 = plt.figure(1, figsize=(12,10), facecolor='w')

	# temperature
	m.pcolormesh(lonm, latm, prop, vmin=hsct[0], vmax=hsct[1],cmap=plt.cm.RdBu_r)
	plt.colorbar(shrink=0.65)

	# velocity
	Q = m.quiver(lonp, latp, u1, v1,zorder=2)
	if lev[kk] > -1000:
		qk = plt.quiverkey(Q, ax1, ay1, 0.2, '0.2m/s', coordinates='data', labelpos='N', zorder=10)
	else:
		qk = plt.quiverkey(Q, ax1, ay1, 0.1, '0.1m/s', coordinates='data', labelpos='N', zorder=10)

	# batimetry
	zb = np.ma.masked_where(h >= lev[kk]*-1, h)
	m.contourf(lonm, latm, zb, colors=('0.7'), alpha=0.5) 
	m.contour(lonm,latm,h, levels = [100], colors = 'k')

	# details
	m.drawcoastlines(zorder=5)
	m.drawcountries(zorder=4)
	m.fillcontinents(color=('0.8'), lake_color='aqua', zorder=3)
	m.drawparallels(np.arange(run.latmin, run.latmax, 1),
		labels=[1, 0, 0, 0], dashes=[1,1000], zorder=6)
	m.drawmeridians(np.arange(run.lonmin, run.lonmax, 2),
		labels=[0, 0, 0, 1], dashes=[1,1000], zorder=7)

	text = 'Z: '+ str(lev[kk]) +' m' 
	plt.text(tx2,ty2,text,color='k',fontsize=10,fontweight='bold')


	string =  "plt.savefig('figures/initial_field/vel-temp_" \
				+ str(lev[kk]) +"m" + ".png')"
	exec(string)
	del hsct, Q, qk
	plt.close()


	# map salinity vs. velocity -----------------------------------------------
	prop = s1.copy()

	fig1 = plt.figure(1, figsize=(12,10), facecolor='w')

	# salinity
	m.pcolormesh(lonm,latm,prop, vmin=hscs[0], vmax=hscs[1],cmap=plt.cm.RdBu_r)
	plt.colorbar(shrink=0.65)

	# velocity
	Q = m.quiver(lonp, latp, u1, v1,zorder=2)
	if lev[kk] > -1000:
		qk = plt.quiverkey(Q,ax1,ay1,0.2,'0.2m/s',coordinates='data',labelpos='N',zorder=10)
	else:
		qk = plt.quiverkey(Q,ax1,ay1,0.1,'0.1m/s',coordinates='data',labelpos='N',zorder=10)

	# batimetry
	zb = np.ma.masked_where(h >= lev[kk]*-1, h)
	m.contourf(lonm, latm, zb, colors=('0.7'), alpha=0.5) 
	m.contour(lonm,latm,h,levels = [100],colors = 'k')

	# details
	m.drawcoastlines(zorder=5)
	m.drawcountries(zorder=4)
	m.fillcontinents(color=('0.8'), lake_color='aqua', zorder=3)
	m.drawparallels(np.arange(run.latmin, run.latmax, 1),
		labels=[1, 0, 0, 0], dashes=[1,1000], zorder=6)
	m.drawmeridians(np.arange(run.lonmin, run.lonmax, 2),
		labels=[0, 0, 0, 1], dashes=[1,1000], zorder=7)

	tx, ty = m( run.lonmin + 0.5, run.latmax - 4 )
	text = 'Z: '+ str(lev[kk]) +' m' 
	plt.text(tx,ty,text,color='k',fontsize=10,fontweight='bold')

	string =  "plt.savefig('figures/initial_field/vel-salt_" \
				+ str(lev[kk]) +"m" + ".png')"
	exec(string)
	del hscs, u1, v1, t1, s1, Q, qk
	plt.close()


# map vbar vs. zeta ----------------------------------------------------------
prop = zeta.copy()

fig1 = plt.figure(1, figsize=(12,10), facecolor='w')

# zeta
m.pcolormesh(lonm,latm,prop, vmin=hscz[0], vmax=hscz[1],cmap=plt.cm.RdBu_r)
plt.colorbar(shrink=0.65)

# velocity
Q = m.quiver(lonp, latp, ubar1, vbar1,zorder=2)
qk = plt.quiverkey(Q,ax1,ay1,0.2,'0.2m/s',coordinates='data',labelpos='N',zorder=10)

# betimetry
zb = np.ma.masked_where(h >= 10, h)
m.contourf(lonm, latm, zb, colors=('0.7'), alpha=0.5) 
m.contour(lonm,latm,h,levels = [100],colors = 'k')

# details
m.drawcoastlines(zorder=5)
m.drawcountries(zorder=4)
m.fillcontinents(color=('0.8'), lake_color='aqua', zorder=3)
m.drawparallels(np.arange(run.latmin, run.latmax, 1),
	labels=[1, 0, 0, 0], dashes=[1,1000], zorder=6)
m.drawmeridians(np.arange(run.lonmin, run.lonmax, 2),
	labels=[0, 0, 0, 1], dashes=[1,1000], zorder=7)

text = '$\eta$ x Vbar' 
plt.text(tx2,ty2,text,color='k',fontsize=10,fontweight='bold')

string =  "plt.savefig('figures/initial_field/vbar-zeta" + ".png')"
exec(string)
plt.close()	
del Q, qk
zi = dz * np.ceil( zt.min() / dz )
zi = np.arange(zi, 0+dz, dz)

# zonal velocity slice -------------------------------------------------------

for ii in range(0,ni.size):
	
	# velocity
	
	xsec = latv[:,0]
	fsec = near(xsec,ni[ii])
	xsec = lonv[0,:]
	z    = np.squeeze( zv[:, fsec, :] )
	prop = np.squeeze( V[: ,fsec , :] ) 	
	xsec.shape = (1, xsec.size)
	x = np.repeat(xsec, 40, axis=0)
	prop = np.ma.masked_where(prop > 1e10, prop)

	# transport calculation
	auxz = np.array(np.gradient(z));auxz = auxz[0,...]; 
	pp = np.where(prop>0);
	transp = (prop[pp]*(dx*111120*np.cos(math.radians(ni[ii])))*auxz[pp]).sum()
	nn = np.where(prop<0);
	transn = (prop[nn]*(dx*111120*np.cos(math.radians(ni[ii])))*auxz[nn]).sum()
	del auxz, pp, nn

	fig1 = plt.figure(1, figsize=(12,6), facecolor='w')
	plt.contourf(x,z,prop, np.arange(vsv[0], vsv[1], vst),cmap=plt.cm.RdBu_r)
	plt.colorbar()
	plt.contour(x,z,prop,levels = [0],colors = 'k')
	CS = plt.contour(x,z,prop, levels = levelsp, colors = 'r')
	plt.clabel(CS,levelsp[::2],inline=1,fmt='%3.1f',fontsize=8)
	CS = plt.contour(x,z,prop,levels = levelsn, colors = 'b')
	plt.clabel(CS,levelsn[::2],inline=1,fmt='%3.1f',fontsize=8)
	plt.axis([run.lonmin, run.lonmax, Zlim[0], Zlim[1]])	    	
	plt.xlabel('Longitude')
	plt.title('Cross-section Velocity, lat = '+ str(round(latv[fsec,0],2)))	
	plt.ylabel('Depth')
	plt.text(x[5,5],Zlim[0]+20,'Transport = '+ "{:5.2f}".format(transp/1000000)+' Sv')
	plt.text(x[5,5],Zlim[0]+70,'Transport = '+ "{:5.2f}".format(transn/1000000)+' Sv')
	plt.text(x[5,5],Zlim[0]+120,'Total transport = '+"{:5.2f}".format((transp+transn)/1000000)+' Sv')
	string =  "plt.savefig('figures/initial_field/vel-zon_" + str(round(latv[fsec,0],2)) + ".png')"
	exec(string)
	plt.close()

	fig1 = plt.figure(1, figsize=(12,6), facecolor='w')
	plt.contourf(x,z,prop, np.arange(vsv[0], vsv[1], vst),cmap=plt.cm.RdBu_r)
	plt.colorbar()
	plt.contour(x,z,prop,levels = [0],colors = 'k')
	CS = plt.contour(x,z,prop, levels = levelsp, colors = 'r')
	plt.clabel(CS,levelsp[::2],inline=1,fmt='%3.1f',fontsize=8)
	CS = plt.contour(x,z,prop,levels = levelsn, colors = 'b')
	plt.clabel(CS,levelsn[::2],inline=1,fmt='%3.1f',fontsize=8)
	plt.axis([run.lonmin, run.lonmax, run.hmax*-1, Zlim[1]])	    	
	plt.xlabel('Longitude')
	plt.title('Cross-section Velocity, lat = '+ str(round(latv[fsec,0],2)))	
	plt.ylabel('Depth')
	plt.text(x[5,5],run.hmax*-1+100,'Transport = '+ "{:5.2f}".format(transp/1000000)+' Sv')
	plt.text(x[5,5],run.hmax*-1+300,'Transport = '+ "{:5.2f}".format(transn/1000000)+' Sv')
	plt.text(x[5,5],run.hmax*-1+500,'Total transport = '+"{:5.2f}".format((transp+transn)/1000000)+' Sv')
	string =  "plt.savefig('figures/initial_field/vel-zon_hmax_" + str(round(latv[fsec,0],2)) + ".png')"
	exec(string)
	plt.close()

	del fsec, z, prop, transp, transn

	# temperature

	xsec = lat[:,0]
	fsec = near(xsec,ni[ii])
	xsec = lonv[0,:]
	z    = np.squeeze( zt[:, fsec, :] )
	prop = np.squeeze( temp[: ,fsec , :] ) 
	xsec.shape = (1, xsec.size)
	x = np.repeat(xsec, 40, axis=0)
	prop = np.ma.masked_where(prop > 1e10, prop)

	fig1 = plt.figure(1, figsize=(12,6), facecolor='w')
	plt.contourf(x,z,prop, np.arange(vsc[0], vsc[1], vst),cmap=plt.cm.Spectral_r)
	plt.colorbar()
	plt.axis([run.lonmin, run.lonmax, run.hmax*-1, Zlim[1]])
	plt.xlabel('Longitude')
	plt.title('Temperature, lat = '+ str(round(lat[fsec,0],2)))	
	plt.ylabel('Depth')
	string =  "plt.savefig('figures/initial_field/temp-zon_" + str(round(lat[fsec,0],2)) + ".png')"
	exec(string)		
	plt.close()
	del fsec, z, prop

del ii

# meridional slices ------------------------------------------------------------- 

for jj in range(0,nj.size):
	
	# velocity
	
	xsec = lonu[0, :]
	fsec = near(xsec,nj[jj])
	xsec = latu[:, 0]
	z    = np.squeeze( zu[:, :, fsec] )
	prop = np.squeeze( U[: ,: , fsec] ) 
	xsec.shape = (1, xsec.size)
	x = np.repeat(xsec, 40, axis=0)
	prop = np.ma.masked_where(prop > 1e10, prop)

	# transport calculation
	auxz = np.array(np.gradient(z));auxz = auxz[0,...];
	pp = np.where(prop>0);
	transp = (prop[pp]*(dx*111120*np.cos(math.radians(x[0,:].mean())))*auxz[pp]).sum()
	nn = np.where(prop<0);
	transn = (prop[nn]*(dx*111120*np.cos(math.radians(x[0,:].mean())))*auxz[nn]).sum()
	del auxz, pp, nn

	fig1 = plt.figure(1, figsize=(12,6), facecolor='w')
	plt.contourf(x,z,prop, np.arange(vsv[0], vsv[1], vst),cmap=plt.cm.RdBu_r)
	plt.colorbar()
	plt.contour(x,z,prop,levels = [0],colors = 'k')
	CS = plt.contour(x,z,prop, levels = levelsp, colors = 'r')
	plt.clabel(CS,levelsp[::2],inline=1,fmt='%3.1f',fontsize=8)
	CS = plt.contour(x,z,prop,levels = levelsn, colors = 'b')
	plt.clabel(CS,levelsn[::2],inline=1,fmt='%3.1f',fontsize=8)
	plt.axis([run.latmin, run.latmax, Zlim[0], Zlim[1]])	    	
	plt.text(x[5,5],Zlim[0]+20,'Transport = '+"{:5.2f}".format(transp/1000000)+' Sv')
	plt.text(x[5,5],Zlim[0]+70,'Transport = '+"{:5.2f}".format(transn/1000000)+' Sv')
	plt.text(x[5,5],Zlim[0]+120,'Total transport = '+"{:5.2f}".format((transp+transn)/1000000)+' Sv')
	plt.xlabel('Latitude')
	plt.title('Cross-section Velocity, lon = '+ str(round(lonu[0,fsec],2)))	
	plt.ylabel('Depth')
	string =  "plt.savefig('figures/initial_field/vel-mer_" + str(round(lonu[0,fsec],2)) + ".png')"
	exec(string)
	plt.close()

	fig1 = plt.figure(1, figsize=(12,6), facecolor='w')
	plt.contourf(x,z,prop, np.arange(vsv[0], vsv[1], vst),cmap=plt.cm.RdBu_r)
	plt.colorbar()
	plt.contour(x,z,prop,levels = [0],colors = 'k')
	CS = plt.contour(x,z,prop, levels = levelsp, colors = 'r')
	plt.clabel(CS,levelsp[::2],inline=1,fmt='%3.1f',fontsize=8)
	CS = plt.contour(x,z,prop,levels = levelsn, colors = 'b')
	plt.clabel(CS,levelsn[::2],inline=1,fmt='%3.1f',fontsize=8)
	plt.axis([run.latmin, run.latmax, run.hmax*-1, Zlim[1]])	    	
	plt.text(x[5,5],run.hmax*-1+100,'Transport = '+"{:5.2f}".format(transp/1000000)+' Sv')
	plt.text(x[5,5],run.hmax*-1+300,'Transport = '+"{:5.2f}".format(transn/1000000)+' Sv')
	plt.text(x[5,5],run.hmax*-1+500,'Total transport = '+"{:5.2f}".format((transp+transn)/1000000)+' Sv')
	plt.xlabel('Latitude')
	plt.title('Cross-section Velocity, lon = '+ str(round(lonu[0,fsec],2)))	
	plt.ylabel('Depth')
	string =  "plt.savefig('figures/initial_field/vel-mer_hmax_" + str(round(lonu[0,fsec],2)) + ".png')"
	exec(string)
	plt.close()

	del fsec, z, prop, transp, transn

	# temperature 

	xsec = lonu[0, :]
	fsec = near(xsec,nj[jj])
	xsec = latu[:, 0]
	z    = np.squeeze( zt[:, :, fsec] )
	prop = np.squeeze( temp[: ,: , fsec] ) 
	   	
	xsec.shape = (1, xsec.size)
	x = np.repeat(xsec, 40, axis=0)
	prop = np.ma.masked_where(prop > 1e10, prop)
	
	fig1 = plt.figure(1, figsize=(12,6), facecolor='w')
	plt.contourf(x,z,prop, np.arange(vsc[0], vsc[1], vst),cmap=plt.cm.Spectral_r)	# meu X-axis tem que ser latitude
	plt.colorbar()
	plt.axis([run.latmin, run.latmax, run.hmax*-1, Zlim[1]])
	plt.xlabel('Latitude')
	plt.title('Temperature, lon = '+ str(round(lonu[0,fsec],2)))	
	plt.ylabel('Depth')
	string =  "plt.savefig('figures/initial_field/temp-mer_" + str(round(lonu[0,fsec],2)) + ".png')"
	exec(string)		
	plt.close()
	del fsec, z, prop

