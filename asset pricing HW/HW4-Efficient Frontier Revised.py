#HW4: Efficient Frontier Revisited

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calc_ex_dev(industry_df, market_df):
    return_set = []
    tracking_error_set = []
    for i in industry_df.columns:
        excess_return = np.mean(industry_df[i]-market_df["Mkt"])
        tracking_error = np.std(industry_df[i]-market_df["Mkt"])
        return_set.append(excess_return)
        tracking_error_set.append(tracking_error)

    expected_deviation = pd.DataFrame({"deviation": return_set, "tracking error": tracking_error_set})
    expected_deviation.index = industry_df.columns
    expected_deviation.to_excel(r"/Users/xiangsihan/Desktop/table1 .xlsx")

def Minimum_Tracking_Error(industry_df, market_df):
    return_set = []
    for i in industry_df.columns:
        excess_return = np.mean(industry_df[i]-market_df["Mkt"])
        return_set.append(excess_return)

    expected_deviation = pd.DataFrame({"deviation": return_set})
    expected_deviation.index = industry_df.columns
    return_dict = {}
    for i in industry_df.columns:
        excess_return = list(industry_df[i]-market_df["Mkt"])
        return_dict[i] = excess_return
    return_mat = pd.DataFrame(return_dict)
    cov = return_mat.cov()
    R_industry, V_industry = np.array(expected_deviation), cov
    print(R_industry)
    print(V_industry)
    Rp = np.linspace(0,0.16,500)
    R, V, e = np.array(R_industry).T, np.array(V_industry), np.ones(len(industry_df.columns))
    V_inv  = np.linalg.inv(V)
    alpha = np.dot(np.dot(R,V_inv), e)
    zeta = np.dot(np.dot(R,V_inv), R.T)
    delta = np.dot(np.dot(e.T,V_inv), e)
    print(alpha,zeta,delta)
    Rmv = alpha/delta #minimum variance
    R_tg = zeta/alpha
    #efficient frontier without riskless asset
    Sigma_portfolio1 = np.sqrt((delta*Rp**2-2*alpha*Rp+zeta)/(zeta*delta-alpha**2))
    print(np.sqrt((delta*Rmv**2-2*alpha*Rmv+zeta)/(zeta*delta-alpha**2)))
    
    Sigma_tg = -np.sqrt(zeta)/(delta*(-Rmv))
    line = (Sigma_tg/R_tg)*Rp

    R_tg1 = float(R_tg[0])
    Sigma_tg1 = float(Sigma_tg[0])
    R_upper = Rp[Rp>Rmv]
    sigma_upper = np.sqrt((delta*R_upper**2-2*alpha*R_upper+zeta)/(zeta*delta-alpha**2))
    
    factor = (np.dot(zeta,delta)-np.dot(alpha,alpha))[0][0]
    zeta1 = zeta[0][0]
    alpha1 = alpha[0]
    a = (np.dot(np.dot(zeta1,V_inv),e)-np.dot(np.dot(alpha1,V_inv),R.T).T)/factor
    b = (np.dot(np.dot(delta,V_inv),R.T).T-np.dot(np.dot(alpha1,V_inv),e))/factor
    w = a.T+b.T*R_tg
    w_df = pd.DataFrame(w).reset_index(drop=True)
    w_df.index = ten_industry.columns
    
    w_df.to_excel(r"table2 .xlsx")

    return Sigma_portfolio1, Rp, line, R_tg1, R_upper, sigma_upper, Sigma_tg1, Sigma_tg, R_tg, Rmv


ten_industry = pd.read_excel(r"Exam_Industries.xlsx",header=0,index_col=0)
market = pd.read_excel(r"Exam_Market.xlsx",header=0,index_col=0)
calc_ex_dev(ten_industry, market)
sigma,Rp, line, R_tg1, R_upper, sigma_upper, Sigma_tg1, Sigma_tg,R_tg, Rmv = Minimum_Tracking_Error(ten_industry,market)


plt.plot(sigma.T, Rp, c = 'b', label = 'Minimum-Tracking-Error Frontier')
plt.yticks(np.arange(0, 0.165, 0.005))
plt.xlabel("Monthly Tracking Error")
plt.ylabel("Expected Monthly Return Deviation (%)")
plt.title("Minimum-Tracking-Error Frontier")
plt.legend()
plt.show()

IR = R_tg1/Sigma_tg1


plt.plot(sigma_upper.T, R_upper, c = "#403990")
plt.plot(line.T,Rp, c = "#CF3D3E")
plt.plot(Sigma_tg, R_tg, 'go', label = 'tangency portfolio', c = "#80A6E2")
plt.yticks(np.arange(0, 0.165, 0.005))
plt.text(Sigma_tg1, R_tg1, f'({Sigma_tg1:.3f}, {R_tg1:.3f})', ha='right', va='bottom', fontsize=12, color='black')
plt.xlabel("Monthly Tracking Error")
plt.ylabel("Expected Monthly Return Deviation (%)")
plt.title("Minimum-tracking-error Frontier")
plt.legend(["minimum-tracking-error frontier",
    "tangent line","tangency portfolio"])


#Monte carlo
def monte_carlo(industry_df,num):
    ret_dot = []
    sigma_dot = []
    for i in range(int(num)): 
        weights = np.random.random((np.shape(ten_industry)[1]))
        total = sum(weights)
        normalized_weights = weights / total
        portfolio_ret = np.sum(normalized_weights*industry_df.mean())
        portfolio_cov = industry_df.cov()
        portfolio_risk = np.sqrt(np.dot(normalized_weights.T, np.dot(portfolio_cov, normalized_weights)))
        ret_dot.append(portfolio_ret)
        sigma_dot.append(portfolio_risk)
    plt.scatter(sigma_dot, ret_dot, marker='o', color='#403990')
    plt.xlabel("std dev of return")
    plt.ylabel("risk premium")
    plt.title("Minimum-Variance Frontier w/o Short Sales")


ten_industry = pd.read_excel(r"/Users/xiangsihan/Desktop/Exam_Industries.xlsx",header=0,index_col=0)

monte_carlo(ten_industry,10e5)


def monte_carlo_uniform(industry_df,num):
    ret_dot1 = []
    sigma_dot1 = []
    for i in range(int(num)): 
        uniform = np.random.uniform(0, 1, (np.shape(ten_industry)[1])) #difference
        weight = 1/uniform
        normalized_weights = weight / np.sum(weight)
        portfolio_ret = np.sum(normalized_weights*industry_df.mean())
        portfolio_cov = industry_df.cov()
        portfolio_risk = np.sqrt(np.dot(normalized_weights.T, np.dot(portfolio_cov, normalized_weights)))
        ret_dot1.append(portfolio_ret)
        sigma_dot1.append(portfolio_risk)
    plt.scatter(sigma_dot1, ret_dot1, marker='o', color='#403990')
    plt.xlabel("std dev of return")
    plt.ylabel("expected return")
    plt.title("Minimum-Variance Frontier(revised) w/o Short Sales")

monte_carlo_uniform(ten_industry,10000)
