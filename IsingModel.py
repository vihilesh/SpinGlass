

import csv
import pprint
import pandas as pd
import numpy as np
from numpy.random import rand, Generator, PCG64, MT19937, SeedSequence
import matplotlib.pyplot as plt
from scipy.sparse import spdiags,linalg,eye
from enum import Enum
import argparse


nt      = 16          #  number of temperature points
N       = 6           #  size of the lattice, N x N
eqSteps = 2**8        #  number of MC sweeps for equilibration
mcSteps = 2**10     #  number of MC sweeps for calculation




class PRNG(Enum):
    PCG64 = 1
    MT19937 = 2

class RNG :
    def __init__(self, RNG):
        if RNG==PRNG.PCG64 :
            self.prng=Generator(PCG64())
        elif RNG ==  PRNG.MT19937 :
            self.prng = Generator(MT19937())

            

def initLattice(RNG, N):

    lattice = RNG.prng.integers(size=(N,N), low=0, high=1, endpoint=True)
    lattice[lattice==0] =- 1

    return lattice  
    

def mcmove(lattice, RNG, beta):
    '''
    Monte Carlo move using Metropolis algorithm 
    '''
    for i in range(N):
        for j in range(N):
                a =  RNG.prng.integers(low=0, high=N, size=1, dtype=np.int8, endpoint=False)
                b =  RNG.prng.integers(low=0, high=N, size=1, dtype=np.int8, endpoint=False)
                s =  lattice[a, b]
                nb = lattice[(a+1)%N,b] + lattice[a,(b+1)%N] + lattice[(a-1)%N,b] + lattice[a,(b-1)%N]
                cost = 2*s*nb
                
                if cost < 0:
                    s *= -1
                elif  RNG.prng.random() < np.exp(-cost*beta):
                    s *= -1
                lattice[a, b] = s    
    return lattice

def calcEnergy(config):
    '''
    Energy of a given configuration
    '''
    energy = 0 
    
    for i in range(len(config)):
        for j in range(len(config)):
            S = config[i,j]
            neighbor = config[(i+1)%N, j] + config[i,(j+1)%N] + config[(i-1)%N, j] + config[i,(j-1)%N]
            energy += -neighbor*S
    return energy/2.  # to compensate for over-counting

def calcMag(config):
    '''
    Magnetization of a given configuration
    '''
    mag = np.sum(config)
    return mag

#switches for different RNG. Make it a command line paramters. 
#rng = RNG(PRNG.PCG64)
rng = RNG(PRNG.MT19937)

parser = argparse.ArgumentParser("IsingModel")
subparser = parser.add_subparsers()
parser_list = subparser.add_parser('RNG choices')
parser_list.add_argument('RNG choices', default='PCG64', const='all', nargs='?', choices=[ 'PCG64', 'MT19937'])
args = parser.parse_args()
pprint(vars(args))

args = parser.parse_args()
print(args.counter + 1)

#get a array of floating temp values between the low and high
T       = np.linspace(1.53, 3.28, nt); 

#zero out the values for E, M, C and X for all temperature points. 
E,M,C,X = np.zeros(nt), np.zeros(nt), np.zeros(nt), np.zeros(nt)


n1, n2  = 1.0/(mcSteps*N*N), 1.0/(mcSteps*mcSteps*N*N) 

#loop over all temperatures in the nt array
for tt in range(nt):
    lattice = initLattice(rng, N)        # initialise

    E1 = M1 = E2 = M2 = 0
    
    
    beta = 1.0/T[tt]
    betasquared = beta * beta     
    
    for i in range(eqSteps):   
       mcmove(lattice, rng, beta)        # equilibrate
     

    for i in range(mcSteps):         
        mcmove(lattice, rng, beta)           
        Ene = calcEnergy(lattice)     # calculate the energy
        Mag = calcMag(lattice)        # calculate the magnetisation

        E1 = E1 + Ene
        M1 = M1 + Mag
        M2 = M2 + Mag*Mag 
        E2 = E2 + Ene*Ene


    # divide by number of sites and iteractions to obtain intensive values    
    E[tt] = n1*E1
    M[tt] = n1*M1
    C[tt] = (n1*E2 - n2*E1*E1)*betasquared
    X[tt] = (n1*M2 - n2*M1*M1)*beta



titlestr = "MC Steps: " + str(mcSteps) + " Eq Steps: " + str(eqSteps) + " Lattice: " + str(N)
f = plt.figure(figsize=(18, 10)); #  
f.suptitle(titlestr)

# does not seem to the most optimal way to drump data to file. This works for now. 
df = pd.DataFrame({"Temperature": T, "Energy": E, "Magnetisation": M, "Specific Heat":C, "Susceptibility": X})
file = str(eqSteps) + "-" + str(mcSteps) + "-" + str(N)
df.to_csv(file+".csv", index=False)

sp =  f.add_subplot(2, 2, 1 );
plt.scatter(T, E, s=50, marker='o', color='IndianRed')
plt.xlabel("Temperature (T)", fontsize=20);
plt.ylabel("Energy ", fontsize=20);         plt.axis('tight');


sp =  f.add_subplot(2, 2, 2 );
plt.scatter(T, abs(M), s=50, marker='o', color='RoyalBlue')
plt.xlabel("Temperature (T)", fontsize=20); 
plt.ylabel("Magnetization ", fontsize=20);   plt.axis('tight');


sp =  f.add_subplot(2, 2, 3 );
plt.scatter(T, C, s=50, marker='o', color='IndianRed')
plt.xlabel("Temperature (T)", fontsize=20);  
plt.ylabel("Specific Heat ", fontsize=20);   plt.axis('tight');   


sp =  f.add_subplot(2, 2, 4 );
plt.scatter(T, X, s=50, marker='o', color='RoyalBlue')
plt.xlabel("Temperature (T)", fontsize=20); 
plt.ylabel("Susceptibility", fontsize=20);   plt.axis('tight');

plt.show()
