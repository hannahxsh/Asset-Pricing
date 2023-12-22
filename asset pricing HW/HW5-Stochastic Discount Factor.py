import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

epsn = np.random.normal(0, 1, 10**4)
nu = []
random_samples = np.random.uniform(0,1,10**4)
for i in random_samples: 
    if i < 0.02: 
        nu.append(np.log(0.6))
    else:
        nu.append(0)
consumption_growth = 0.02+0.02*epsn+nu

#power_utility = 
std = []
mean = []
M = []
gamma = np.linspace(0,4,1000)
for i in gamma:
    std_m = np.std(0.99*np.exp(i*(1-consumption_growth)))
    mean_m = np.mean(0.99*np.exp(i*(1-consumption_growth)))
    m_value = std_m/mean_m
    std.append(std_m)
    M.append(m_value)

plt.style.use("ggplot")
mpl.rcParams["axes.grid"] = True
mpl.rcParams["grid.color"] = "grey"
mpl.rcParams["grid.alpha"] = 0.25
mpl.rcParams["axes.facecolor"] = "white"



data_df = pd.DataFrame({"gamma":gamma, "M":M})
gamma_04 = data_df[data_df["M"]>0.4].iloc[0]["gamma"]
M_04 = data_df[data_df["M"]>0.4].iloc[0]["M"]

plt.plot(gamma, M, c = "#403990", label = "H-J bound")
plt.plot(gamma_04, M_04, 'go', c = "#CF3D3E", label = "smallest value")
plt.text(gamma_04, M_04, f'({gamma_04:.1f}, {M_04:.2})', ha='right', va='bottom', fontsize=12, color='black')
plt.xticks(np.arange(0.0,4.1, 0.1))
#plt.yticks(np.arange(0,0.6, 0.05))
plt.xlabel('b0')
plt.ylabel('sigma/mu')
plt.legend()
plt.title("Hansen-Jagannathan Bound")
