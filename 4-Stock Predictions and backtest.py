import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from finrl.env import env_stocktrading
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

###########################################################################################################################
#RETRIEVING AND PROCESSING THE DATA
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


#SPLITTING DATA IN TRAIN AND TEST

train = data_split(df, '2009-01-01', '2019-01-01')
trade = data_split(df, '2019-01-01', '2020-12-01')

#Set trade evn
stock_dimension = len(df.tic.unique())
state_space=stock_dimension
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

#GET BASELINE FOR BACKTEST AND COMPUTE ITS STATS
baseline_df = get_baseline(
        ticker='^DJI', start='2019-01-01', end='2020-12-01'
    )

baseline_returns = get_daily_return(baseline_df, value_col_name="close")
perf_stats_all = perf_func( returns=baseline_returns,
                              factor_returns=baseline_returns,
                                positions=None, transactions=None, turnover_denom="AGB")
#MAKE PREDICTIONS AND BACKTEST MODEL A2C

model_prova= A2C.load("C:/Users/federico/Documents/Desktop\Tesi/trained_models/A2CProva")
df_daily_return, df_actions = DRLAgent.DRL_prediction(model=model_prova,
                        environment = e_trade_gym)

DRL_strat = backtest.convert_daily_return_to_pyfolio_ts(df_daily_return)
perf_func = timeseries.perf_stats
perf_stats_all = perf_func( returns=DRL_strat,
                              factor_returns=DRL_strat,
                                positions=None, transactions=None, turnover_denom="AGB")
perf_stats_all.to_csv("C:/Users/federico/Documents/Desktop/Tesi/results/stats/statistiche_a2c_prova.csv")

with pyfolio.plotting.plotting_context(font_scale=1.1):
    pyfolio.create_full_tear_sheet(returns=DRL_strat,
                                   benchmark_rets=baseline_returns)

plt.savefig("results/results_a2c_prova.png")

#MAKE PREDICTIONS AND BACKTEST MODEL PPO

model_prova= PPO.load("C:/Users/federico/Documents/Desktop\Tesi/trained_models/PPOProva")
df_daily_return, df_actions = DRLAgent.DRL_prediction(model=model_prova,
                        environment = e_trade_gym)
DRL_strat = backtest.convert_daily_return_to_pyfolio_ts(df_daily_return)
perf_func = timeseries.perf_stats
perf_stats_all = perf_func( returns=DRL_strat,
                              factor_returns=DRL_strat,
                                positions=None, transactions=None, turnover_denom="AGB")
perf_stats_all.to_csv("C:/Users/federico/Documents/Desktop/Tesi/results/stats/statistiche_ppo_prova.csv")

with pyfolio.plotting.plotting_context(font_scale=1.1):
    pyfolio.create_full_tear_sheet(returns=DRL_strat,
                                   benchmark_rets=baseline_returns)

plt.savefig("results/results_ppo_prova.png")


#MAKE PREDICTION AND BACKTEST MODEL DDPG

model_prova= DDPG.load("C:/Users/federico/Documents/Desktop\Tesi/trained_models/DDPGProva")
df_daily_return, df_actions = DRLAgent.DRL_prediction(model=model_prova,
                        environment = e_trade_gym)
DRL_strat = backtest.convert_daily_return_to_pyfolio_ts(df_daily_return)
perf_func = timeseries.perf_stats
perf_stats_all = perf_func( returns=DRL_strat,
                              factor_returns=DRL_strat,
                                positions=None, transactions=None, turnover_denom="AGB")
perf_stats_all.to_csv("C:/Users/federico/Documents/Desktop/Tesi/results/stats/statistiche_ddpg_prova.csv")

with pyfolio.plotting.plotting_context(font_scale=1.1):
    pyfolio.create_full_tear_sheet(returns=DRL_strat,
                                   benchmark_rets=baseline_returns)

plt.savefig("results/results_ddpg_prova.png")


#MAKE PREDICTION AND BACKTEST MODEL DDPG

model_prova= SAC.load("C:/Users/federico/Documents/Desktop\Tesi/trained_models/SACProva")
df_daily_return, df_actions = DRLAgent.DRL_prediction(model=model_prova,
                        environment = e_trade_gym)
DRL_strat = backtest.convert_daily_return_to_pyfolio_ts(df_daily_return)
perf_func = timeseries.perf_stats
perf_stats_all = perf_func( returns=DRL_strat,
                              factor_returns=DRL_strat,
                                positions=None, transactions=None, turnover_denom="AGB")
perf_stats_all.to_csv("C:/Users/federico/Documents/Desktop/Tesi/results/stats/statistiche_sac_prova.csv")

with pyfolio.plotting.plotting_context(font_scale=1.1):
    pyfolio.create_full_tear_sheet(returns=DRL_strat,
                                   benchmark_rets=baseline_returns)

plt.savefig("results/results_sac_prova.png")

