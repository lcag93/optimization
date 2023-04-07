#-------------------------------------------------------------------------------
# Name:        SCE_Python_shared version -  helpfunctions
# This is the implementation for the SCE algorithm,
# written by Q.Duan, 9/2004 - converted to python by Van Hoey S. 2011
# Purpose:
#
# Author:      VHOEYS
#
# Created:     11/10/2011
# Copyright:   (c) VHOEYS 2011
# Licence:     <your licence>
#-------------------------------------------------------------------------------
#!/usr/bin/env python

import os
import sys
import numpy as np
import math

################################################################################
##  Sampling called from SCE
################################################################################

def SampleInputMatrix(nrows,npars,bu,bl,iseed,distname='randomUniform'):
    '''
    Create inputparameter matrix for nrows simualtions,
    for npars with bounds ub and lb (np.array from same size)
    distname gives the initial sampling ditribution (currently one for all parameters)

    returns np.array
    '''
    np.random.seed(iseed)
    x=np.zeros((nrows,npars))
    bound = bu-bl
    for i in range(nrows):
        # x[i,:]= bl + DistSelector([0.0,1.0,npars],distname='randomUniform')*bound  #only used in full Vhoeys-framework
        x[i,:]= bl + np.random.rand(1,npars)*bound
    return x

################################################################################
##   FUNCTION CALL FROM SCE-ALGORITHM !!
################################################################################

def constraint(x):
    return x[0]*x[1]*x[2]*x[3] - 25.0

def play(x):
    '''
    This is the play function
    '''
    
    if constraint(x)>=0.0:
        try:
            x[2] = math.sqrt(40.0 - x[0]**2 - x[1]**2 - x[3]**2)
        except:
            return 1000
        return x[0]*x[3]*(x[0]+x[1]+x[2])+x[2]
    else:
        return 1000
    
def rosenbrock(x):
    '''
    This is the Rosenbrock Function
    Bound: X1=[-10,10], X2=[-10,10]; Global Optimum: 0.0, (1,1)
    '''
    
    return (100.0*(x[1]-x[0]**2))**2 + (1.0-x[0])**2

def griewank(x):
    '''
    This is the Griewank Function (2-D)
    Bound: X(i)=[-100,100], for i=1,2
    Global Optimum: 0, at origin
    '''
    
    return (x[0]**2+x[1]**2)/4000.0 - np.cos(x[0]/np.sqrt(1))*np.cos(x[1]/np.sqrt(2)) + 1

def constraint1(x):
    return math.prod(x) - 0.75

def constraint2(x):
    return sum(x) - 7.5*len(x)

def bump(x):
    '''
    This is the Bump Function (N-D)
    Bound: X(i)=[-1,1], for i=1,2,...,N
    Global Optimum: ??, at origin
    '''
    
    n = len(x)
    sumsqrt = 0
    for i in range(n):
        sumsqrt += x[i]**2
    if np.sqrt(sumsqrt)<1:
        return -(np.exp(-1/(1-sumsqrt)))
    else:
        return 0

def keane(x):
    '''
    This is the Keane Function (8-D)
    Bound: X(i)=[0,10], for i=1,2,...,8
    Global Optimum: 0, at origin
    '''
    
    n = len(x)
    sum1 = 0.0
    prod = 1.0
    sum2 = 0.0
    for i in range(n):
        sum1 += np.cos(x[i])**4
        prod *= np.cos(x[i])**2
        sum2 += i*(x[i]**2)
    if constraint1(x)>=0 and constraint2(x)<=0:
        return -np.abs(sum1-2*prod)/np.sqrt(sum2)
    else:
        return 0

def EvalObjF(npar,x,testnr=4,data=None):
    if testnr==1:
        return rosenbrock(x)
    elif testnr==2:
        return griewank(x)
    elif testnr==3:
        return bump(x)
    else:
        return play(x)
    
