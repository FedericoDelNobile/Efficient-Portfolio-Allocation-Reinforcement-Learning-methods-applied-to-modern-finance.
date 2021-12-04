import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from finrl.env import env_stocktrading

matplotlib.use('Agg')
import datetime

from finrl.config import config
from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import data_split
from finrl.env.environment import EnvSetup
from finrl.env.env_portfolio import StockPortfolioEnv
from finrl.env.EnvMultipleStock_train import StockEnvTrain
from finrl.env.EnvMultipleStock_trade import StockEnvTrade
from finrl.model.models import DRLAgent
from finrl.trade import backtest
import pyfolio
import os
from pyfolio import timeseries
###########################################################################################################################
#RETRIEVE AND PREPROCESS DATA
########################################################################################################################

df = YahooDownloader(start_date = '2009-01-01',
                  end_date = '2020-12-01',
                  ticker_list = config.DOW_30_TICKER).fetch_data()

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

train = data_split(df, '2009-01-01', '2019-01-01')
trade = data_split(df, '2019-01-01', '2020-12-01')
stock_dimension = len(trade.tic.unique())

state_space = stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")


#SETTING THE ENV

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
env_trade =e_trade_gym.get_sb_env()


##########################################################################################################################
#STARTING GRIDSEARCH
##########################################################################################################################

#Save models parameters for later
dic={}

#Set Grid and agent
PPO_PARAMS = {
    "n_steps": [1024,2048,3072],
    "ent_coef": [0.005,0.007],
    "learning_rate": [0.0001,0.0005],
    "batch_size": [128,256]
}
agent_ppo = DRLAgent(env=env_train)

#Train the models
for n in PPO_PARAMS["n_steps"]:
    for  rate in PPO_PARAMS["learning_rate"]:
        for coef in PPO_PARAMS["ent_coef"]:
            for size in PPO_PARAMS["batch_size"]:
                ppo_params={"n_steps": n, "ent_coef": coef, "learning_rate": rate, "batch_size": size }
                nome= str(ppo_params["n_steps"]) + "_" + str(ppo_params["ent_coef"]) + "_" + str(ppo_params["learning_rate"])
                # Train models inside the loop
                model_ppo = agent_ppo.get_model("ppo", model_kwargs=ppo_params)
                trained_ppo = agent_ppo.train_model(model=model_ppo,
                                                    tb_log_name='ppo',
                                                    total_timesteps=80000)
                #make predictions
                df_daily_return, df_actions = DRLAgent.DRL_prediction(model=trained_ppo,
                                                                      environment=e_trade_gym)

                #Evaluate strategy
                DRL_strat = backtest.convert_daily_return_to_pyfolio_ts(df_daily_return)
                perf_func = timeseries.perf_stats
                perf_stats_all = perf_func(returns=DRL_strat,
                                           factor_returns=DRL_strat,
                                           positions=None, transactions=None, turnover_denom="AGB")
                if max==0:
                    max= perf_stats_all["Cumulative returns"]*perf_stats_all["Sharpe ratio"]
                else:
                    challenger= perf_stats_all["Cumulative returns"]*perf_stats_all["Sharpe ratio"]
                    if challenger>max:
                        max=challenger
                        best_model= trained_ppo
                dic[nome] = [perf_stats_all["Annual return"], perf_stats_all["Cumulative returns"],perf_stats_all["Sharpe ratio"]]



# Saving Results##########################################################################################
models= pd.DataFrame(dic)
models.to_csv("C:/Users/federico/Documents/Desktop/Tesi/results/stats/Grid_search_Baseline_2.csv")


########################################################################################################################################
# CHOOSING THE BEST MODEL#
####################################################################################################################################
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
#Cumulative Ret* Sharpe
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

#################### Training the best model and getting its stats###########################
agent_ppo = DRLAgent(env=env_train)
ppo_params={"n_steps": 1024, "ent_coef": 0.005, "learning_rate": 0.0001, "batch_size": 256}

model_ppo = agent_ppo.get_model("ppo", model_kwargs=ppo_params)
trained_ppo = agent_ppo.train_model(model=model_ppo,tb_log_name='ppo',total_timesteps=80000)
df_daily_return, df_actions = DRLAgent.DRL_prediction(model=trained_ppo,environment=e_trade_gym)
DRL_strat = backtest.convert_daily_return_to_pyfolio_ts(df_daily_return)
perf_func = timeseries.perf_stats
perf_stats_all = perf_func(returns=DRL_strat,
                                       factor_returns=DRL_strat,
                                       positions=None, transactions=None, turnover_denom="AGB")
perf_stats_all.to_csv("C:/Users/federico/Documents/Desktop/Tesi/results/stats/Baseline_best_model_2.csv")

########### Get Baselines ###############################################
baseline_df = backtest.get_baseline(
        ticker='^DJI', start='2019-01-01', end='2020-12-01'
    )

baseline_returns = backtest.get_daily_return(baseline_df, value_col_name="close")

########## Plots ##########################################
with pyfolio.plotting.plotting_context(font_scale=1.1):
    pyfolio.create_full_tear_sheet(returns=DRL_strat,
                                   benchmark_rets=baseline_returns)

plt.savefig("results/results_best_baseline_2.png")
