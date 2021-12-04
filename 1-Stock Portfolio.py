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
from pyfolio import timeseries
import os
#######################################################################################################################
#SETTING DIRECTORIES
###########################################################################################################################
if not os.path.exists("./" + config.DATA_SAVE_DIR):
    os.makedirs("./" + config.DATA_SAVE_DIR)
if not os.path.exists("./" + config.TRAINED_MODEL_DIR):
    os.makedirs("./" + config.TRAINED_MODEL_DIR)
if not os.path.exists("./" + config.TENSORBOARD_LOG_DIR):
    os.makedirs("./" + config.TENSORBOARD_LOG_DIR)
if not os.path.exists("./" + config.RESULTS_DIR):
    os.makedirs("./" + config.RESULTS_DIR)
##########################################################################################################################
#RETRIEVING AND PROCESSING THE DATA
########################################################################################################################
df = YahooDownloader(start_date = '2009-01-01',
                  end_date = '2020-12-01',
                  ticker_list = config.DOW_30_TICKER).fetch_data()
#Add Thecnical Indicators
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

#######################################################################################################################
#TRAINING THE MODELS
########################################################################################################################
#1 A2C
agent = DRLAgent(env = env_train)

A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.005, "learning_rate": 0.0002}
model_a2c = agent.get_model(model_name="a2c",model_kwargs = A2C_PARAMS)

trained_a2c = agent.train_model(model=model_a2c,
                                tb_log_name='a2c',
                                total_timesteps=60000)
trained_a2c.save(f"C:/Users/federico/Documents/Desktop/Tesi/trained_models/A2CProva")
######################################################################################################################
#2 PPO

agent_ppo = DRLAgent(env = env_train)
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.005,
    "learning_rate": 0.0001,
    "batch_size": 128,
}
model_ppo = agent_ppo.get_model("ppo",model_kwargs = PPO_PARAMS)

trained_ppo = agent_ppo.train_model(model=model_ppo,
                             tb_log_name='ppo',
                             total_timesteps=80000)

trained_ppo.save(f"C:/Users/federico/Documents/Desktop/Tesi/trained_models/PPOProva")
#######################################################################################################################
#3 DDPG

agent_ddpg = DRLAgent(env = env_train)
DDPG_PARAMS = {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001}


model_ddpg = agent_ddpg.get_model("ddpg",model_kwargs = DDPG_PARAMS)

trained_ddpg = agent_ddpg.train_model(model=model_ddpg,
                             tb_log_name='ddpg',
                             total_timesteps=50000)
trained_ddpg.save(f"C:/Users/federico/Documents/Desktop/Tesi/trained_models/DDPGProva")
#######################################################################################################################
#4 SAC
agent_sac = DRLAgent(env = env_train)
SAC_PARAMS = {
    "batch_size": 128,
    "buffer_size": 100000,
    "learning_rate": 0.0003,
    "learning_starts": 100,
    "ent_coef": "auto_0.1",
}

model_sac = agent_sac.get_model("sac",model_kwargs = SAC_PARAMS)
trained_sac = agent_sac.train_model(model=model_sac,
                             tb_log_name='sac',
                             total_timesteps=50000)
trained_sac.save(f"C:/Users/federico/Documents/Desktop/Tesi/trained_models/SACProva")