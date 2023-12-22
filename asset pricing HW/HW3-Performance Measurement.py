#HW3: Performance Measurement


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression



def calc_measurement(Rf_df, industry_df):

    Rf = Rf_df["Rf"]

#calculate beta
    beta_set = []
    alpha_set = []
    market_premium = np.array(Rf_df["Rm-Rf"])
    SMB = np.array(Rf_df["SMB"])
    HML = np.array(Rf_df["HML"])
    independent_variable = np.array([market_premium,SMB,HML])
    for i in industry_df.columns:
    #Calculate excess return
        excess_return = np.array(industry_df[i]-Rf)
    #regression
        model = LinearRegression().fit(market_premium.reshape(-1,1),excess_return)
        three_factor_model = LinearRegression().fit(independent_variable.T,excess_return)
        beta_set.append(model.coef_[0])
        #3factor alpha
        alpha_set.append(three_factor_model.intercept_)

    k=0
    sharpe_set = []
    sortino_set = []
    treynor_set = []
    jensen_set = []
    for i in industry_df.columns:
        Ri = industry_df[i]
        x = Ri-Rf
        semi_var = np.sum((x[x<0])**2)/len(x)
        #Calculate Sharpe Ratio
        sharpe_ratio = (Ri-Rf).mean()/(Ri-Rf).std()
        #Calculate Sortino Ratio (Target is risk-free rate)
        sortino_ratio = (Ri-Rf).mean()/np.sqrt(semi_var)
        #Calculate Treynor Ratio
        treynor_ratio = (Ri-Rf).mean()/beta_set[k]
        #Calculate Jensen Alpha
        jensen_alpha = (Ri-Rf).mean()-beta_set[k]*(risk_factor["Rm-Rf"].mean())
    
        k=k+1
    
        sharpe_set.append(sharpe_ratio)
        sortino_set.append(sortino_ratio)
        treynor_set.append(treynor_ratio)
        jensen_set.append(jensen_alpha)


    data = {
        "Industry name": ten_industry.columns,
        "Sharpe Ratio": sharpe_set,
        "Sortino Ratio": sortino_set,
        "Treynor Ratio": treynor_set,
        "Jensen Alpha": jensen_set,
        "3-factor alpha": alpha_set
    }
    Performance = pd.DataFrame(data)
    return Performance

ten_industry = pd.read_excel(r"Industry_Portfolios.xlsx",header=0,index_col=0)
risk_factor = pd.read_excel(r"Risk_Factors.xlsx",header=0,index_col=0)

Performance = calc_measurement(risk_factor, ten_industry)



def draw_pic(str):
    plt.bar(Performance["Industry name"], Performance[str], color = "#403990")
    plt.title(str)
    plt.xlabel('Industry Name')
    plt.ylabel(str)
    for i, value in enumerate((Performance[str])):
        plt.text(i, value+0.008, f'{value:.2f}', ha='center', va='center_baseline',fontsize=10)
    plt.show()

draw_pic("Sortino Ratio")
