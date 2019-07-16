#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import time

##############################################
# edit these options:                        #
case = 'upwelling'                           #
##############################################

print(' \n')
print('Case: ', case)
print(' \n')

data = np.loadtxt("../output/{}_ek.out".format(case))

# Tentar pegar esse dt automaticamente do .log utilizando o .bash
#dt = 2880 # 300 seconds = DT on the .in
dt = int(data[-1,0]) # timestep da simulação
t = data[:,0]
t = (t*dt)/86400

# Pegando o valor da energia cinética do .log
ek = data[:,7]

plt.figure(figsize=(8,5))
plt.title('Total Ek temporal evolution ({} timesteps)'.format(dt), fontsize=16)
plt.grid(color='lightgray')
plt.plot(t, ek, 'k')
plt.xlabel('Days of simulation', fontsize=16)
plt.ylabel('J m$^{-2}$', fontsize=16)
#plt.axis([0,30,0,0.01])
#plt.show()
plt.savefig("figures/results/{}_{}_timesteps_{}.png".format(case, dt, time.time()))
plt.close()
