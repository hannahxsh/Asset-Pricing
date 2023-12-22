import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
gs = np.exp(0.02+0.02*np.random.normal(0,1,int(10**4)))

def cal_E(x,gs):
    summation = 0
    for g in gs:
        if x*g >= 1.0303:
            summation += x*g - 1.0303
        else:
            summation += 2*(x*g - 1.0303)
    mean = summation/len(gs)
    return mean

#e(x) = 0.99*b_0*E(v(x*g)) + 0.99*x - 1
def cal_x(b0, x1, x2):
    
    x = (x1+x2)/2
    
    while(abs(0.99*b0*cal_E(x, gs) + 0.99*x - 1))>(1e-5):
        if (0.99*b0*cal_E(x, gs) + 0.99*x - 1)>0:
            x2 = x
        else:
            x1 = x
        x = (x1+x2)/2
        
    return x        

df_ans = pd.DataFrame([i / 10.0 for i in range(4)], columns=["b0"])

lst_x = []
for b0 in df_ans["b0"]:
    x1 = 1
    x2 = 1.1
    lst_x.append(cal_x(b0, x1, x2))

df_ans["x"] = lst_x

df_ans["price_dividend_ratio"] = 1/(df_ans["x"]-1)

df_ans["E_Rm"] = df_ans["x"]*np.exp(0.0202)

df_ans["ERP"] = df_ans["E_Rm"] - 1.0303

plt.style.use("ggplot")
plt.plot(df_ans["b0"], df_ans["price_dividend_ratio"])
plt.xlabel("b0")
plt.ylabel("price-dividend ratio")
#plt.xticks(df_ans["b0"],rotation=45,fontsize=9)
plt.title("price-dividend ratio vs b0")
plt.show()


plt.xlabel("b0")
plt.ylabel("equity premium")
plt.title("equity premium vs b0")
plt.plot(df_ans["b0"],df_ans["ERP"])
plt.show()
