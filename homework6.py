#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 00:34:59 2017

@author: Kaina
"""
import numpy as np
import math
import scipy.integrate as integrate
import matplotlib.pyplot as plt

pi = math.pi
d = 1.0
q = int(math.pow(10, 6))


F1 = lambda x: abs(x - math.pi/10)
F2 = lambda x: (x-1)**2
F3 = lambda x: (math.e)**x
F4 = lambda x: (math.pow(math.sin(x),2))

resultf1 = integrate.quad(F1, -1, 1)
resultf2 = integrate.quad(F2, -1, 1)
resultf3 = integrate.quad(F3, -1, 1)
resultf4 = integrate.quad(F4, -pi, pi)


def rightSums(f, a, b, n):
    h = float(b - a) / n
    s = 0.0
    i = 1.0
    while (i<=(n)):
        s = f(a+h*i)*h + s
        i=i+1
    return s
    
def trapezoidal(f, a, b, n):
    h = float(b - a) / n
    s = 0.0
    s += f(a)/2.0
    for i in range(1, n):
        s += f(a + i*h)
    s += f(b)/2.0
    return s * h

def simpson(f, a, b, n):
    h = float(b - a) / n
    s = f(a) + f(b)
    for i in range(1, n, 2):
        s += 4 * f(a + i * h)
    for i in range(2, n-1, 2):
        s += 2 * f(a + i * h)
    return s * h /3


'''print("simpson F1:    ", simpson(F1, -1, 1, 10))
print("simpson F2:    ", simpson(F2, -1, 1, 10))
print("simpson F3:    ", simpson(F3, -1, 1, 10))
print("resultf1: ", resultf1)
print("resultf2: ", resultf2)
print("resultf3: ", resultf3)'''
#print("resultf4: ", resultf4)


N = [10, 100, 1000, 10000, 100000, 1000000]
RightSumsf1 = []
RightSumsf2 = []
RightSumsf3 = []
Trapezoidf1 = []
Trapezoidf2 = []
Trapezoidf3 = []
Simpsonf1 = []
Simpsonf2 = []
Simpsonf3 = []

RightSumsf4 = []
Trapezoidf4 = []
Simpsonf4 = []

for i in N:
    RightSumsf1.append(abs(rightSums(F1, -1, 1, i)-resultf1[0]))
    RightSumsf2.append(abs(rightSums(F2, -1, 1, i)-resultf2[0]))
    RightSumsf3.append(abs(rightSums(F3, -1, 1, i)-resultf3[0]))
    Trapezoidf1.append(abs(trapezoidal(F1, -1, 1, i)-resultf1[0]))
    Trapezoidf2.append(abs(trapezoidal(F2, -1, 1, i)-resultf2[0]))
    Trapezoidf3.append(abs(trapezoidal(F3, -1, 1, i)-resultf3[0]))
    Simpsonf1.append(abs(simpson(F1, -1, 1, i)-resultf1[0]))
    Simpsonf2.append(abs(simpson(F2, -1, 1, i)-resultf2[0]))
    Simpsonf3.append(abs(simpson(F3, -1, 1, i)-resultf3[0]))
    
    RightSumsf4.append(abs(rightSums(F4, -pi, pi, i)))
    Trapezoidf4.append(abs(trapezoidal(F4, -pi, pi, i)))
    Simpsonf4.append(abs(simpson(F4, -pi, pi, i)))
    
print RightSumsf4
print Trapezoidf4
print Simpsonf4



logN = [math.log(i) for i in N]
F1RSErrors = [math.log(i) for i in RightSumsf1]
F2RSErrors = [math.log(i) for i in RightSumsf2]
F3RSErrors = [math.log(i) for i in RightSumsf3]
F1TRAPErrors = [math.log(i) for i in Trapezoidf1]
F2TRAPErrors = [math.log(i) for i in Trapezoidf2]
F3TRAPErrors = [math.log(i) for i in Trapezoidf3]
F1SIMErrors = [math.log(i) for i in Simpsonf1]
F2SIMErrors = [math.log(i) for i in Simpsonf2]
F3SIMErrors = [math.log(i) for i in Simpsonf3]



r = np.polyfit(logN, F1RSErrors, 1)[0]
s = np.polyfit(logN, F2RSErrors, 1)[0]
t = np.polyfit(logN, F3RSErrors, 1)[0]
u = np.polyfit(logN, F1TRAPErrors, 1)[0]
v = np.polyfit(logN, F2TRAPErrors, 1)[0]
w = np.polyfit(logN, F3TRAPErrors, 1)[0]
x = np.polyfit(logN, F1SIMErrors, 1)[0]
y = np.polyfit(logN, F2SIMErrors, 1)[0]
z = np.polyfit(logN, F3SIMErrors, 1)[0]


'''print ('alpha for Right Sums F1   : ', -r)
print ('alpha for Right Sums F2   : ', -s)
print ('alpha for Right Sums F3   : ', -t)
print ('alpha for Trapezoid F1    : ', -u)
print ('alpha for Trapezoid F2    : ', -v)
print ('alpha for Trapezoid F3    : ', -w)
print ('alpha for Simpson F1      : ', -x)
print ('alpha for Simpson F2      : ', -y)
print ('alpha for Simpson F3      : ', -z)



'''
'''

plt.figure(1,figsize=(20, 15))
plt.subplot(222)
plt.title('Right Sums')
plt.plot(N, RightSumsf1, label = 'F1')
plt.plot(N, RightSumsf2, label = 'F2')
plt.plot(N, RightSumsf3, label = 'F3')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('log(N)')
plt.ylabel('log(E)')
plt.legend()
plt.grid(True)
plt.show()


plt.figure(2,figsize=(20, 15))
plt.subplot(222)
plt.title('Trapezoid')
plt.plot(N, Trapezoidf1, label = 'F1')
plt.plot(N, Trapezoidf2, label = 'F2')
plt.plot(N, Trapezoidf3, label = 'F3')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('log(N)')
plt.ylabel('log(E)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(3,figsize=(20, 15))
plt.subplot(222)
plt.title('Simpson"s Method')
plt.plot(N, Simpsonf1, label = 'F1')
plt.plot(N, Simpsonf2, label = 'F2')
plt.plot(N, Simpsonf3, label = 'F3')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('log(N)')
plt.ylabel('log(E)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(4,figsize=(20, 15))
plt.subplot(222)
plt.title('Sin(x)^2')
plt.plot(N, RightSumsf4, label = 'Right Sums')
plt.plot(N, Trapezoidf4, label = 'Trapezoid')
plt.plot(N, Simpsonf4, label = 'Simpson')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('log(N)')
plt.ylabel('log(E)')
plt.legend()
plt.grid(True)
plt.show()'''