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
from _datetime import datetime
import pyfolio
from pyfolio import timeseries
from finrl.trade import backtest

#########################################################################################################################
#RETRIEVE DATA
##########################################################################################################################

lista=["XLM-USD","ADA-USD","TRX-USD","XMR-USD","DASH-USD","NEO-USD","BTC-USD","ETH-USD","BCH-USD","LTC-USD","EOS-USD"]
df = YahooDownloader(start_date = '2015-01-01',
                  end_date = '2021-04-01',
                  ticker_list = lista).fetch_data()

df = df.sort_values(['date', 'tic'], ignore_index=True)
#Taking the df from the point in wich every crypto is traded#

newdf= df.truncate(6378)
df= newdf

#########################################################################################################################
#Feature Engeneering
##########################################################################################################################

df = FeatureEngineer().preprocess_data(df.copy())

# add covariance matrix as states
df = df.sort_values(['date', 'tic'], ignore_index=True)
df.index = df.date.factorize()[0]
cov_list = []
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

#SPLITTING DATASET

df.index = df.date.factorize()[0]

train = data_split(df, '2018-06-10', '2020-06-09')
trade = data_split(df, '2020-06-10', '2021-03-31')

#SETTING THE ENVIRORMENT

stock_dimension = len(trade.tic.unique())

state_space = stock_dimension

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

#######################################################################################
######### Find Best Model####################################################
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

models= pd.DataFrame(dic)
models.to_csv("C:/Users/federico/Documents/Desktop/Tesi/results/stats/Grid_search_Crypto.csv")


##########################################################################################
############ Training the best model #############################################

a2c_params={"n_steps": 5, "ent_coef": 0.005, "learning_rate": 0.0001}
model_a2c = agent_a2c.get_model("a2c", model_kwargs=a2c_params)
trained_a2c = agent_a2c.train_model(model=model_a2c, tb_log_name='a2c', total_timesteps=60000)

#######################################################################################
####### Evaluating perforances ########################################################
###################################
####### Getting baselines       ####################################################


baseline2= pd.read_csv("C:/Users/federico/Documents/Desktop/Tesi/venv/df/CMC_ALL_CRYPTO.csv")
prova=baseline2["Date"].tolist()
baseline_date=[]
for el in prova:
    el=el[:-12]
    el=datetime.strptime(el, '%B %d %Y')
    baseline_date.append(el)
data= pd.DataFrame({"Date_Format":baseline_date})
baseline2= pd.concat([baseline2,data],axis=1)
baseline2=baseline2.drop(['Date','BTC', 'Unnamed: 3'], axis=1)
baseline2.rename(columns={'CCMIX':"daily_return",'Date_Format':"date"}, inplace=True)
baseline2=baseline2.truncate(before=1257)
baseline2=backtest.convert_daily_return_to_pyfolio_ts(baseline2)
baseline_returns= baseline2.pct_change()
############################################################################################
############ Get Predictions and Baseline Stats ###################################################################

df_daily_return, df_actions = DRLAgent.DRL_prediction(model=trained_a2c,
                        environment = e_trade_gym)
DRL_strat = backtest.convert_daily_return_to_pyfolio_ts(df_daily_return)
perf_func = timeseries.perf_stats
            perf_stats_all = perf_func(returns=DRL_strat,
                                       factor_returns=DRL_strat,
                                       positions=None, transactions=None, turnover_denom="AGB")

############### PLOT #######################################################
with pyfolio.plotting.plotting_context(font_scale=1.1):
    pyfolio.create_full_tear_sheet(returns=DRL_strat,
                                   benchmark_rets=baseline_returns)


plt.savefig("results/results_crypto_best_1.png")

#get statistics################################################
perf_func = timeseries.perf_stats
perf_stats_all = perf_func(returns=DRL_strat,factor_returns=DRL_strat,positions=None, transactions=None, turnover_denom="AGB")
perf_stats_all.to_csv("results/crypto_best_stats.csv")

######################################################################################################################
#GRID SEARCH FOR DDPG
#NOT MENTIONED IN THE PAPER AS IT DID NOT PRODUCE APPRECIABLE RESULTS
######################################################################################################################
dic={}

#Setting params

agent_ddpg = DRLAgent(env = env_train)

DDPG_PARAMS = {"batch_size": 128, "buffer_size": [50000,25000,75000], "learning_rate": [0.001,0.003,0.005]}

for  size in DDPG_PARAMS["buffer_size"]:
    for rate in DDPG_PARAMS["learning_rate"]:
        ddpg_params={"batch_size": 128, "buffer_size": size, "learning_rate": rate}
        nome= str(ddpg_params["batch_size"]) + "_" + str(ddpg_params["learning_rate"]) + "_" + str(ddpg_params["buffer_size"])

        # Train models inside the loop
        model_ddpg = agent_ddpg.get_model("ddpg", model_kwargs=ddpg_params)
        trained_ddpg= agent_ddpg.train_model(model=model_ddpg,
                                            tb_log_name='ddpg',
                                            total_timesteps=50000)
        #make predictions
        df_daily_return, df_actions = DRLAgent.DRL_prediction(model=trained_ddpg,
                                                              environment=e_trade_gym)

        #Evaluate strategy
        DRL_strat = backtest.convert_daily_return_to_pyfolio_ts(df_daily_return)
        perf_func = timeseries.perf_stats
        perf_stats_all = perf_func(returns=DRL_strat,
                                   factor_returns=DRL_strat,
                                   positions=None, transactions=None, turnover_denom="AGB")
        dic[nome]=[perf_stats_all["Annual return"], perf_stats_all["Cumulative returns"], perf_stats_all["Sharpe ratio"]]

# Saving Results##########################################################################################
best_model.save(f"C:/Users/federico/Documents/Desktop/Tesi/trained_models/BEST_DDPG_CRYPTO")
models= pd.DataFrame(dic)
models.to_csv("C:/Users/federico/Documents/Desktop/Tesi/results/stats/Grid_search_crypto _2.csv")

