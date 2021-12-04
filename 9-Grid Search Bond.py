import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import datetime
from finrl.config import config
from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import data_split
import os
from finrl.env.env_portfolio import StockPortfolioEnv
from finrl.model.models import DRLAgent
from pyfolio import timeseries
from finrl.trade import backtest
import pyfolio
from finrl.trade import backtest

###########################################################################################################################
#RETRIEVE AND PREPROCESS DATA
########################################################################################################################

lista=["VCIT","LQD","VCSH","IGSB","IGIB","SPSB","FLOT","SPIB","USIG","VCLT","ICSH","GSY","IGLB","SLQD"]
df = YahooDownloader(start_date = '2010-01-01',
                  end_date = '2021-04-01',
                  ticker_list = lista).fetch_data()

df = df.sort_values(['date', 'tic'], ignore_index=True)

df.to_csv("C:/Users/federico/Documents/Desktop/Tesi/venv/df/BondDF.csv")

colonna= df["tic"].tolist()
indici=[]
for el in lista:
    indici.append(colonna.index(el))
newdf= df.truncate(11624)
df= newdf


df = FeatureEngineer().preprocess_data(df.copy())

# add covariance matrix as states
df = df.sort_values(['date', 'tic'], ignore_index=True)
df.index = df.date.factorize()[0]

cov_list = []
# look back is one year
lookback = 252
for i in range(lookback, len(df.index.unique())):
    data_lookback = df.loc[i - lookback:i, :]
    price_lookback = data_lookback.pivot_table(index='date', columns='tic', values='close')
    return_lookback = price_lookback.pct_change().dropna()
    covs = return_lookback.cov().values
    cov_list.append(covs)

df_cov = pd.DataFrame({'date': df.date.unique()[lookback:], 'cov_list': cov_list})
df = df.merge(df_cov, on='date')
df = df.sort_values(['date', 'tic']).reset_index(drop=True)
df.head()


#Splitting and train
train = data_split(df, '2014-12-16', '2018-12-04')
trade = data_split(df, '2018-12-06', '2021-03-31')
stock_dimension = len(trade.tic.unique())

state_space = stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")


#SETTING ENVs
env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "transaction_cost_pct": 0.001,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4

}

e_train_gym = StockPortfolioEnv(df=train, **env_kwargs)
env_train, _ = e_train_gym.get_sb_env()
e_trade_gym = StockPortfolioEnv(df=trade, **env_kwargs)


###########################################################################################################################
#STARTING GRIDSEARCH#
##########################################################################################################################

#GENERATING THE GRID #############################################################################################

A2C_PARAMS = {"n_steps": [5,7,3,9],
              "ent_coef": [0.005,0.007,0.002],
              "learning_rate": [0.0002,0.0004,0.0001]}

#GRID SEARCH ALGORITHM ############################################################################################

agent_a2c = DRLAgent(env=env_train)
dic={}

#create grid
for n in A2C_PARAMS["n_steps"]:
    for  rate in A2C_PARAMS["learning_rate"]:
        for coef in A2C_PARAMS["ent_coef"]:
            a2c_params={"n_steps": n, "ent_coef": coef, "learning_rate": rate}
            nome= str(a2c_params["n_steps"]) + "_" + str(a2c_params["learning_rate"]) + "_" + str(a2c_params["ent_coef"])

            # Train models inside the loop
            model_a2c = agent_a2c.get_model("a2c", model_kwargs=a2c_params)
            trained_a2c = agent_a2c.train_model(model=model_a2c,
                                                tb_log_name='a2c',
                                                total_timesteps=60000)
            #make predictions
            df_daily_return, df_actions = DRLAgent.DRL_prediction(model=trained_a2c,
                                                                  environment=e_trade_gym)

            #Evaluate strategy
            DRL_strat = backtest.convert_daily_return_to_pyfolio_ts(df_daily_return)
            perf_func = timeseries.perf_stats
            perf_stats_all = perf_func(returns=DRL_strat,
                                       factor_returns=DRL_strat,
                                       positions=None, transactions=None, turnover_denom="AGB")
            dic[nome]=[perf_stats_all["Annual return"], perf_stats_all["Cumulative returns"], perf_stats_all["Sharpe ratio"]]
models= pd.DataFrame(dic)
models.to_csv("C:/Users/federico/Documents/Desktop/Tesi/results/stats/Grid_search_Bond.csv")

#Annual Ret * Sharpe
l=[]
for el in dic.keys():
    l.append([dic[el][1]*dic[el][2], el])
Massimo=0
for el in l:
    if el[0]>Massimo:
        Massimo=el[0]
for el in l:
    if el[0]== Massimo:
        print(el[1])
        print(Massimo)
#Cumulative Ret * Sharpe
l=[]
for el in dic.keys():
    l.append([dic[el][0]*dic[el][2], el])
Massimo=0
for el in l:
    if el[0]>Massimo:
        Massimo=el[0]
for el in l:
    if el[0]== Massimo:
        print(el[1])
        print(Massimo)

##########################################################################################
############ Training the best model #############################################

a2c_params={"n_steps": 5, "ent_coef": 0.002, "learning_rate": 0.0001}
model_a2c = agent_a2c.get_model("a2c", model_kwargs=a2c_params)
trained_a2c = agent_a2c.train_model(model=model_a2c,tb_log_name='a2c',total_timesteps=60000)

#######################################################################################
####### Evaluating perforances ########################################################
###################################
####### Getting baselines and baseline stats  ####################################################

baseline_df = backtest.get_baseline(
        ticker='AGG', start='2018-12-06', end='2021-03-31'
    )
baseline_returns = backtest.get_daily_return(baseline_df, value_col_name="close")
perf_func = timeseries.perf_stats
perf_stats_all = perf_func(returns=baseline_returns,
                                       factor_returns=baseline_returns,
                                       positions=None, transactions=None, turnover_denom="AGB")

########## Predictions ##############################################################

df_daily_return, df_actions = DRLAgent.DRL_prediction(model=trained_a2c,
                                                      environment=e_trade_gym)
DRL_strat = backtest.convert_daily_return_to_pyfolio_ts(df_daily_return)

with pyfolio.plotting.plotting_context(font_scale=1.1):
    pyfolio.create_full_tear_sheet(returns=DRL_strat,
                                   benchmark_rets=baseline_returns)


plt.savefig("results/results_bond_best.png")



######################################################################################################################
#GRID SEARCH 2 SAC
#NOT INCLUDED IN THE PAPER AS IT DID NOT PRODUCED APPRECIABLE RESULTS
######################################################################################################################
agent_sac = DRLAgent(env = env_train)
SAC_PARAMS = {
    "batch_size": [128,256],
    "buffer_size": [100000,50000,25000],
    "learning_rate": [0.0003,0.0001,0.0005],
    "learning_starts": 100,
    "ent_coef": "auto_0.1",
}
# Create dic for saving stats and variable to select best model
dic={}

#create grid
for size in SAC_PARAMS["batch_size"]:
    for  buffer in SAC_PARAMS["buffer_size"]:
        for rate in SAC_PARAMS["learning_rate"]:
            sac_params={"batch_size": size, "buffer_size": buffer, "learning_rate": rate, "learning_starts": 100, "ent_coef": "auto_0.1"}
            nome= str(sac_params["batch_size"]) + "_" + str(sac_params["buffer_size"]) + "_" + str(sac_params["learning_rate"])

            # Train models inside the loop
            model_sac = agent_sac.get_model("sac", model_kwargs=sac_params)
            trained_sac = agent_sac.train_model(model=model_sac,
                                                tb_log_name='sac',
                                                total_timesteps=50000)
            #make predictions
            df_daily_return, df_actions = DRLAgent.DRL_prediction(model=trained_sac,
                                                                  environment=e_trade_gym)

            #Evaluate strategy
            DRL_strat = backtest.convert_daily_return_to_pyfolio_ts(df_daily_return)
            perf_func = timeseries.perf_stats
            perf_stats_all = perf_func(returns=DRL_strat,
                                       factor_returns=DRL_strat,
                                       positions=None, transactions=None, turnover_denom="AGB")
            dic[nome]=[perf_stats_all["Annual return"], perf_stats_all["Cumulative returns"], perf_stats_all["Sharpe ratio"]]




models= pd.DataFrame(dic)
models.to_csv("C:/Users/federico/Documents/Desktop/Tesi/results/stats/Grid_search_Bond_SAC.csv")

