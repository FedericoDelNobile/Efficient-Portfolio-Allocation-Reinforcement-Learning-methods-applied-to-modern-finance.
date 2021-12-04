import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from finrl.env import env_stocktrading
import datetime
from datetime import datetime
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
from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from stable_baselines3 import SAC
from finrl.trade import backtest
from pyfolio import timeseries
from finrl.trade.backtest import get_baseline
from finrl.trade.backtest import get_daily_return
import pyfolio
from IPython.display import display, HTML
#########################################################################################################################
#RETRIEVE AND PREPROCESS DATA
########################################################################################################################
lista=["XLM-USD","ADA-USD","TRX-USD","XMR-USD","DASH-USD","NEO-USD","BTC-USD","ETH-USD","BCH-USD","LTC-USD","EOS-USD"]
df = YahooDownloader(start_date = '2015-01-01',
                  end_date = '2021-04-01',
                  ticker_list = lista).fetch_data()

df = df.sort_values(['date', 'tic'], ignore_index=True)

colonna= df["tic"].tolist()
indici=[]
for el in lista:
    indici.append(colonna.index(el))
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

e_trade_gym = StockPortfolioEnv(df = trade, **env_kwargs)



#GET BASELINES
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

#BASELINE STATS
perf_func = timeseries.perf_stats
perf_stats_all_baseline = perf_func( returns=baseline_returns,
                              factor_returns=baseline_returns,
                                positions=None, transactions=None, turnover_denom="AGB")
#MAKE PREDICTIONS AND BACKTEST MODEL A2C

model_prova= A2C.load("C:/Users/federico/Documents/Desktop\Tesi/trained_models/A2C_Crypto")
df_daily_return, df_actions = DRLAgent.DRL_prediction(model=model_prova,
                        environment = e_trade_gym)
DRL_strat = backtest.convert_daily_return_to_pyfolio_ts(df_daily_return)
perf_func = timeseries.perf_stats
perf_stats_all = perf_func( returns=DRL_strat,
                              factor_returns=DRL_strat,
                                positions=None, transactions=None, turnover_denom="AGB")
perf_stats_all.to_csv("C:/Users/federico/Documents/Desktop/Tesi/results/stats/statistiche_a2c_crypto.csv")

with pyfolio.plotting.plotting_context(font_scale=1.1):
    pyfolio.create_full_tear_sheet(returns=DRL_strat,
                                   benchmark_rets=baseline_returns)

plt.savefig("results/results_a2c_crypto.png")

#MAKE PREDICTIONS AND BACKTEST MODEL PPO

model_prova= PPO.load("C:/Users/federico/Documents/Desktop\Tesi/trained_models/PPOCrypto")
df_daily_return, df_actions = DRLAgent.DRL_prediction(model=model_prova,
                        environment = e_trade_gym)
DRL_strat = backtest.convert_daily_return_to_pyfolio_ts(df_daily_return)
perf_func = timeseries.perf_stats
perf_stats_all = perf_func( returns=DRL_strat,
                              factor_returns=DRL_strat,
                                positions=None, transactions=None, turnover_denom="AGB")
perf_stats_all.to_csv("C:/Users/federico/Documents/Desktop/Tesi/results/stats/statistiche_ppo_crypto.csv")

with pyfolio.plotting.plotting_context(font_scale=1.1):
    pyfolio.create_full_tear_sheet(returns=DRL_strat,
                                   benchmark_rets=baseline_returns)

plt.savefig("results/results_ppo_crypto.png")


#MAKE PREDICTION AND BACKTEST MODEL DDPG

model_prova= DDPG.load("C:/Users/federico/Documents/Desktop\Tesi/trained_models/DDPGCrypto")
df_daily_return, df_actions = DRLAgent.DRL_prediction(model=model_prova,
                        environment = e_trade_gym)
DRL_strat = backtest.convert_daily_return_to_pyfolio_ts(df_daily_return)
perf_func = timeseries.perf_stats
perf_stats_all = perf_func( returns=DRL_strat,
                              factor_returns=DRL_strat,
                                positions=None, transactions=None, turnover_denom="AGB")
perf_stats_all.to_csv("C:/Users/federico/Documents/Desktop/Tesi/results/stats/statistiche_ddpg_crypto.csv")

with pyfolio.plotting.plotting_context(font_scale=1.1):
    pyfolio.create_full_tear_sheet(returns=DRL_strat,
                                   benchmark_rets=baseline_returns)

plt.savefig("results/results_ddpg_crypto.png")


#MAKE PREDICTION AND BACKTEST MODEL SAC

model_prova= SAC.load("C:/Users/federico/Documents/Desktop\Tesi/trained_models/SACCrypto ")
df_daily_return, df_actions = DRLAgent.DRL_prediction(model=model_prova,
                        environment = e_trade_gym)
DRL_strat = backtest.convert_daily_return_to_pyfolio_ts(df_daily_return)
perf_func = timeseries.perf_stats
perf_stats_all = perf_func( returns=DRL_strat,
                              factor_returns=DRL_strat,
                                positions=None, transactions=None, turnover_denom="AGB")
perf_stats_all.to_csv("C:/Users/federico/Documents/Desktop/Tesi/results/stats/statistiche_sac_crypto.csv")

with pyfolio.plotting.plotting_context(font_scale=1.1):
    pyfolio.create_full_tear_sheet(returns=DRL_strat,
                                   benchmark_rets=baseline_returns)

plt.savefig("results/results_sac_crypto.png")

