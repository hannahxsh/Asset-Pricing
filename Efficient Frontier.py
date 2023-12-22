#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 16:43:58 2023

@author: xiangsihan
"""
#HW1: Efficient frontier


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def Mean_Variance_Frontier(r):
    
    R_industry, STD_industry, V_industry = r.mean(), r.std(), r.cov()
    print(R_industry)
    print(STD_industry)
    print(V_industry)
    (pd.DataFrame({"mean": R_industry, "standard deviation": STD_industry})).to_excel(r"/Users/xiangsihan/Desktop/table .xlsx")
    R_portfolio = np.linspace(0,2.5,201)
    R, V, e = np.array(R_industry).T, np.array(V_industry), np.ones(len(r.columns))
    V_inv  = np.linalg.inv(V)
    alpha = np.dot(np.dot(R.T,V_inv),e)
    zeta = np.dot(np.dot(R.T,V_inv),R)
    delta = np.dot(np.dot(e.T,V_inv),e)
    Rmv = alpha/delta #minimum variance
    #efficient frontier without riskless asset
    Sigma_portfolio1 = np.sqrt((delta*R_portfolio**2-2*alpha*R_portfolio+zeta)/(zeta*delta-alpha**2))
    
    #efficient frontier with risklesss asset
    Rf = 0.13
    Sigma_portfolio2 = np.sqrt((R_portfolio-Rf)**2/(zeta-2*alpha*Rf+delta*Rf**2))
    R_portfolio2 = R_portfolio[R_portfolio>Rf]
    Sigma_portfolio2 = Sigma_portfolio2[R_portfolio>Rf]
    
    #tangency portfolio variance
    R_tangency = (alpha*Rf-zeta)/(delta*Rf-alpha)
    Sigma_tangency = -(np.sqrt(zeta-2*alpha*Rf+delta*Rf**2))/(delta*(Rf-alpha/delta))
    
    #calculate weight for each industry at tangency point
    a = (zeta*np.dot(V_inv,e)-alpha*np.dot(V_inv,R))/(zeta*delta-alpha**2)
    b = (delta*np.dot(V_inv,R)-alpha*np.dot(V_inv,e))/(zeta*delta-alpha**2)
    w = a + b*R_tangency
    print(w)
    return R_portfolio, R_portfolio2, Sigma_portfolio1, Sigma_portfolio2, R_tangency, Sigma_tangency, alpha, zeta, delta, Rmv

r = pd.read_excel('/Users/xiangsihan/Desktop/Exam_Industries.xlsx',header=0,index_col=0)
R_p, R_p2, Sigma_p_nrl, Sigma_p_rl, R_t, Sigma_t, alpha, zeta, delta, Rmv = Mean_Variance_Frontier(r)

R_efficient = R_p[R_p>=Rmv]
R_inefficient = R_p[R_p<Rmv]
sigma1 = np.sqrt((delta*R_efficient**2-2*alpha*R_efficient+zeta)/(zeta*delta-alpha**2))
sigma2 = np.sqrt((delta*R_inefficient**2-2*alpha*R_inefficient+zeta)/(zeta*delta-alpha**2))
#efficient frontier


fig,ax = plt.subplots()
plt.xlim(0,5)

plt.plot(sigma1, R_efficient, c = "b", label = 'Efficient Froutier')
plt.plot(sigma2, R_inefficient, linestyle='--', c = "b",  label = 'Inefficient Froutier')
plt.plot(Sigma_p_rl, R_p2, c = 'r', label = 'CAL')
plt.plot(Sigma_t, R_t, 'go', label = 'tangency portfolio')
plt.text(Sigma_t, R_t, f'({Sigma_t:.2f}, {R_t:.2f})', ha='right', va='bottom', fontsize=12, color='black')
plt.xlabel("std dev of return")
plt.xlim(0,10)
plt.ylabel("expected return")
plt.title("Efficient Frontier with riskless asset")
plt.legend()
plt.show()

#command+1
plt.plot(Sigma_p_nrl, R_p, c = 'b')
plt.yticks(np.arange(0, 2.6, 0.1))
plt.xlabel("standard deviation of (monthly) return")
plt.ylabel("expected (monthly) return(%)")
plt.title("Minimum-Variance Without Riskless Asset")
plt.show()





