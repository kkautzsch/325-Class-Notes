#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 13:34:07 2019

@author: carlkrutz
"""

#1.1536 0.058555 0.87885

import math
import numpy as np 
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

def pullData(dataSet = 'lab8_data3.txt'):
    f = open(dataSet) 
    data = []
    for line in f:
        data_line = line.rstrip().split(' ')
        data.append(data_line)
    xVals = []
    yVals = []
    yError = []
    for line in data: 
        xVals.append(float(line[0]))
        yVals.append(float(line[1]))
        yError.append(float(line[2]))
    return [xVals, yVals, yError]

def plotData(dataset = 'lab8_data3.txt'):
    data = pullData(dataset)
    x = data[0]
    y = data[1]
    yerr = data[2]
    plt.errorbar(x, y, yerr = yerr, fmt = '*', markersize = 4)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    
def plotLogData(dataset = 'lab8_data3.txt'):
    data = pullData(dataset)
    x = data[0]
    y = np.log(data[1])
    yerr = np.array(data[2])/np.array(data[1])
    plt.errorbar(x, y, yerr = yerr, fmt = '*', markersize = 4)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

def plotLogLogData(dataset = 'lab8_data3.txt'):
    data = pullData(dataset)
    x = np.log(data[0])
    y = np.log(data[1])
    yerr = np.array(data[2])/np.array(data[1])
    plt.errorbar(x, y, yerr = yerr, fmt = '*', markersize = 4)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
#plotData()


# our nonlinear example function (4 fit parameters)
def expFunc(x, a, b, c, d):
    return a * np.exp(-b * x) + c

def quadFunc(x, a, b, c, d):
    return a*x*x + b*x + c

def linFunc(x, a, b, c, d):
    return a*x+b

def cubFunc(x, a, b, c, d):
    return a*x*x*x+b*x*x+c*x+d

def logFunc(x, a, b, c, d):
    return a*np.log(b*x)+c

def invFunc(x, a, b, c, d):
    return a/x+b

def powFunc(x, a, b, c, d):
    return a*x**b + c

def bestFit(func = powFunc, dataset = 'lab8_data3.txt'):
    fullData = pullData(dataset)
    xdata = fullData[0]
    ydata = fullData[1]
    errY = fullData[2]
    plt.errorbar(xdata, ydata, yerr = errY, fmt='o', label='data', markersize = 4)

    # do the nonlinear fit (to the true function) using scipy.optimize.curve_fit
    popt, pcov = curve_fit(func, xdata, ydata, sigma = errY, absolute_sigma=True)
    print("Best fit parameters: ", popt)
    print("Errors on fit parameters: ", np.sqrt(np.diag(pcov)))

    plt.plot(xdata, func(xdata, popt[0]*np.ones(len(xdata)),
                     popt[1]*np.ones(len(xdata)),
                     popt[2]*np.ones(len(xdata)),
                     popt[3]*np.ones(len(xdata))), 'r-',
                    label='fit: a=%5.3f, b=%5.3f, c=%5.3f, d=%5.3f' % tuple(popt))

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(fontsize = 11)
    plt.show()
    return xdata, ydata, errY, popt
    
def chisq(func = powFunc, dataset = 'lab8_data3.txt'):
    xdata, ydata, errY, popt = bestFit(func, dataset)
    chisqVal = 0
    for i in range(0,len(xdata)):
        chisqVal = chisqVal + (func(xdata[i], popt[0], popt[1],popt[2],
                     popt[3]) - ydata[i])**2/(errY[i]**2) 
    return chisqVal/(len(xdata)-3)








