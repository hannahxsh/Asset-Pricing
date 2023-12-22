#HW2: Capital Asset Pricing Model
#Fitting SML

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def calc_alpha_beta_CAPM(market_df, industry_df):
### Market model
#Calculate market premium
    market_premium = np.array(market_df["Mkt"]).reshape(-1, 1) #List name could change
    alpha_set = []
    beta_set = []
    for i in industry_df.columns:
    #Calculate excess return
        excess_return = np.array(industry_df[i])
    #regression
        model = LinearRegression().fit(market_premium,excess_return)
        alpha_set.append(model.intercept_)
        beta_set.append(model.coef_[0])
    return alpha_set, beta_set



# Industry_name = list(ten_industry.columns)
# market_model = pd.DataFrame(alpha_set,beta_set).reset_index()
# market_model.index = Industry_name
# market_model.columns = ["beta", "alpha"]
# market_model.to_excel(r'C:\Users\shixi\Desktop\market_model.xlsx')

###Security Market Line


def SML(market_df, industry_df):
    mean_industry = industry_df.mean()
    mean_market = market_df.mean()
    var_market = market_df.std()**2
    def calc_cov(value):
        return np.cov(value,market_df["Mkt"])[0, 1]
    cov = industry_df.apply(calc_cov)
    beta = cov/var_market[0]
    beta_reg = np.append(beta,1).reshape(-1, 1)
    expected_return = np.append(mean_industry,mean_market)
    SML = LinearRegression().fit(beta_reg,expected_return)

    intercept_SML = SML.intercept_
    slope_SML = SML.coef_
    f = lambda x: slope_SML[0]*x + intercept_SML
    x = np.array([0,2])
    plt.style.use("ggplot")
    plt.figure(figsize=(8, 6))
    plt.plot(x,f(x),c='#403990',label='Security Market Line')
    plt.ylim(0,0.015)
    plt.scatter(beta,mean_industry,c='#CF3D3E',label='Industry Portfolios')
    plt.scatter(1,mean_market,c='#80A6E2',label='Market Portfolio')
    plt.xlabel("Beta")
    plt.ylabel("expected return")
    plt.title("Security Market Line")
    plt.legend()



#/100 mens %
ten_industry = pd.read_excel(r"/Users/xiangsihan/Desktop/Exam_Industries.xlsx",header=0,index_col=0)/100
market = pd.read_excel(r"/Users/xiangsihan/Desktop/Exam_Market.xlsx",header=0,index_col=0)/100
alpha, beta = calc_alpha_beta_CAPM(market, ten_industry)
SML(market, ten_industry)

ten_industry.columns
