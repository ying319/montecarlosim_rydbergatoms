# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:15:15 2019

@author: vcpq38
"""
from __future__ import division
import numpy
from arc import *
import matplotlib.pyplot as pyplot
import csv
from datetime import datetime
from pandas import DataFrame  

'''

def make_transrate_LUT(atom_type, n_max = 70, temp = 350, save = True): 
Makes a look-up table (LUT) of the transition rates for all possible transitions up to a certain value of n. 
The states and transition rates are returned from the function or are saved as a .csv file.
    atom_dict = {'Cs':Caesium(), 'Rb87':Rubidium87()}
    atom = atom_dict[atom_type]
    ground_dict = {'Cs':([6,0,0.5]), 'Rb87':([5,0,0.5])}
    ground_state = ground_dict[atom_type]

    states = atom.extraLevels[:]
    states.append(ground_state)
    # First create a list of all states within the atom up to n=n_max with l<=5
    for n in range(atom.groundStateN,int(n_max+1)):
        for l in range(0,5):
            if l==0:
                new_state = [n,l,0.5]
                if new_state not in states:
                    states.append(new_state)
            else:
                new_state = [n,l,l-0.5]
                if new_state not in states:
                    states.append(new_state)
                new_state = [n,l,l+0.5]
                if new_state not in states:
                    states.append(new_state)
    print ('Total states: {}'.format(int(len(states))))
    

    transitionRates = np.zeros((len(states),len(states)))
    # Now calculate the transition rate between every pair of states
    for i in range(len(states)):
        print ('Calculating transitions for state {} of {}'.format(i+1, len(states)))
        for j in range(len(states)):
            n1,l1,j1=states[i]
            n2,l2,j2=states[j]
            if abs(l2-l1)==1:
                if abs(j2-j1)<2-1e-8:
                    #print n1,l1,j1,n2,l2,j2
                    transitionRates[i,j] = atom.getTransitionRate(n1,l1,j1,n2,l2,j2,temperature=temp)
    if save:
        table = numpy.hstack((states, transitionRates))
        # This will save the look-up table file in the current working directory
        numpy.savetxt('Trans_Rates_nmax={}_temp={}K_{}.csv'.format(int(n_max), int(temp), atom_type), table, delimiter = ',')
    return states, transitionRates



file = 'Trans_Rates_nmax=70_temp=300K_Rb87.csv'
'''
def get_rates_from_LUT(file):
    '''Retrieves the state and transition rate data from a previously created look-up table.'''
    LUT = numpy.genfromtxt(file, delimiter = ',')
    states = LUT[:,:3]
    rates = LUT[:,3:]
    print (states.shape, rates.shape)
    return states, rates
    
def calculate_spectrum(states, transitionRates, n1,l1,j1, atom_type = 'Cs', iters = 300000, 
spectrum_range = (400,750), spectrum_resolution = 0.5):
    '''Uses a Monte-Carlo approach to simulate fluorescence from a given atomic state.
    Takes previously calculated arrays of all possible states and transition rates and a target state.
    The spectrum_range argument specifies the wavelength range of fluorescence to record. 
    Fluorescence at other wavelengths is discarded.
    Returns bin edges and a histogram of emitted wavelengths'''
    atom_dict = {'Cs':Caesium(), 'Rb87':Rubidium87()}
    atom = atom_dict[atom_type]
    ground_dict = {'Cs':([6,0,0.5]), 'Rb87':([5,0,0.5])}
    ground_state = ground_dict[atom_type]
    starting_index = np.where(np.isclose(states,[n1,l1,j1]).all(axis=1))[0][0]#??
    emitted_wavelengths = []
    for i in range(iters):
        state_index = starting_index
        # Loop condition is true until the atomic ground state is reached
        while True:
            n1,l1,j1 = states[state_index]
            rates = transitionRates[state_index,:] # gets the information for transition rates out of the current state
            probs = rates/sum(rates) # turns transition rates into transition probabilities
            probs_cum = np.cumsum(probs) # calculates cumulative probability #??
            new_state_index = np.where(probs_cum > np.random.random())[0][0] #use random number to select new state based on cumulative transition probability
            n2,l2,j2 = states[new_state_index]
            # calculate the transition wavelength between the initial and randomly chosen state and append that to the list of emitted wavelengths
            emitted_wavelengths.append(-atom.getTransitionWavelength(int(n1),int(l1),j1,int(n2),int(l2),j2))
            state_index = new_state_index
            if (n2==ground_state[0]) & (l2==ground_state[1]):
                # breaks the loop if the randomly selected state is the atomic ground state
                break

    emitted_wavelengths = np.array(emitted_wavelengths) *1e9 #turn the wavelength list into an array with wavelengths in nm
    hist,bin_edges = numpy.histogram(emitted_wavelengths, int((spectrum_range[1] - spectrum_range[0])/spectrum_resolution),spectrum_range)
    # create a histogram of emitted wavelengths within the specified range
    hist = numpy.array(hist,dtype=float)
    # bin edges is the lower bound on the wavelength bins of the emission histogram
    # hist/iters gives the probability of emission in the wavelength range specified by the bin edges
    return bin_edges[:-1], hist/iters
#starting_index, emitted_wavelengths



if __name__ == "__main__":
    #states, rates = make_transrate_LUT('Rb87', n_max = 70, temp = 300)
    # the look-up table only needs to be created once, it is time consuming!
    # After it is created and saved it can be called by the 'get_rates_from_LUT' function
    # Uncomment the line below to use a previously saved LUT, the path may need correcting
    a = []
    b = []
    c = [] 
    d = [] 
    e = []
    f = [] 
    g = []
    h = []
    i = []
    j = []
    start=datetime.now()

    #i=0
    atom = Caesium()
    states, rates = get_rates_from_LUT(r'C:\Users\vcpq38\OneDrive - Durham University\code\laptop\Trans_Rates_nmax=70_temp=300K_Cs.csv')
    for n in range(10,30,1):
        for n1 in range(10,30,1): 
            for l in range(0,3,1): 
                for j1 in numpy.arange(1/2, 3/2+1, 1):
                    for j2 in numpy.arange(l-1/2,l+1/2+1, 1):
                        thz_off_state = [n,1,j1]
                        thz_on_state = [n1,l,abs(j2)] # Refers to state 13_D_5/2 (S = 0, P = 1 etc)
            #print(atom.getTransitionWavelength(n,2,5/2, n1,1,3/2))
                        #if abs(j1-j2)==0 or abs(j1-j2)==1:
                        print('rydsate', thz_off_state)                            
                        print('thzstate', thz_on_state)
                        thz = abs(atom.getTransitionFrequency(n,1,j1, n1,l,j2)/10**12) #THz
                        print('thz fre', thz)
                        dm_ryd = abs(atom.getDipoleMatrixElement(7, 0, 0.5, 0.5, n,1,j1, 0.5, 0))
                        dm_thz = abs(atom.getDipoleMatrixElement( n,1,j1,0.5, n1,l,abs(j2), 0.5, 0))
                        print('DM_ryd', dm_ryd)
                        print('DM_Thz', dm_thz)
                        if thz >= 0.1 and thz <= 2 and dm_ryd != 0 and dm_thz!= 0:
                            wvls, probs_on = calculate_spectrum(states, rates, *thz_on_state)
                            wvls, probs_off = calculate_spectrum(states, rates, *thz_off_state)
                            rdb_wavelength = abs(atom.getTransitionWavelength(n,1,j1, 7,0,1/2))
                            num1 = numpy.where(wvls == wvls[numpy.where(probs_on == max(probs_on))]-5)[0]
                            num2 = numpy.where(wvls == wvls[numpy.where(probs_on == max(probs_on))]+5)[0]
                            probon = sum(probs_on[int(num1):int(num2)])#max(probs_on)
                            proboff = sum(probs_off[int(num1):int(num2)]) #probs_off[numpy.where(probs_on == max(probs_on))]
                            ratio = probon/proboff
                            radi_wvl = wvls[numpy.where(probs_on == max(probs_on))] #nm                                                        
                            print('rdb_wavelength', rdb_wavelength)                            
                            print('radiate wavelength', radi_wvl)
                            print('max_probon', probon)                            
                            print(ratio)
                            
                            a.append(thz)
                            b.append(rdb_wavelength)
                            c.append(ratio)
                            d.append(radi_wvl)
                            e.append(thz_off_state)
                            f.append(thz_on_state)                          
                            g.append(dm_ryd)
                            h.append(dm_thz)
                            i.append(probon)
                            j.append(proboff)
                            
                            pyplot.figure()
                            pyplot.plot(wvls,probs_on, label = 'THz on')
                            pyplot.plot(wvls, probs_off, alpha = 0.75, label = 'THz off')
                            pyplot.legend(loc=0)
                            pyplot.title('%s - %s_%s nm.png' %(thz_off_state, thz_on_state, radi_wvl))
                            pyplot.xlabel('Fluorescence Wavelength (nm)(300000iterations)')
                            pyplot.ylabel('Prob. of emission')
                            pyplot.savefig("C:/Users/vcpq38/OneDrive - Durham University/code/laptop/1211_Cs/%s - %s_%s nm.png" %(thz_off_state, thz_on_state, radi_wvl), dpi=300, bbox_inches='tight')
                            pyplot.show()

    spectro = {'thz': list(a),'rdb_wavelength': list(b), 'radi_wvl':list(d), 'thz_off_state':list(e), 'thz_on_state':list(f), 'dm_ryd':list(g), 'dm_thz':list(h), 'probon':list(i), 'proboff': list(j), 'ratio': list(c)}
    df = DataFrame(spectro)
    export = df.to_csv('C:/Users/vcpq38/OneDrive - Durham University/code/laptop/1211_Cs/Csspectrom.csv')
    
    print(datetime.now()-start)


