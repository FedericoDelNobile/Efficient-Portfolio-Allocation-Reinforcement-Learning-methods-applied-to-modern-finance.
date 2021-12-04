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
#########################################################################################################################
#RETRIEVE AND PREPROCESS DATA
########################################################################################################################
#Crypto Portfolio Ticker List
lista=["XLM-USD","ADA-USD","TRX-USD","XMR-USD","DASH-USD","NEO-USD","BTC-USD","ETH-USD","BCH-USD","LTC-USD","EOS-USD"]
df = YahooDownloader(start_date = '2015-01-01',
                  end_date = '2021-04-01',
                  ticker_list = lista).fetch_data()

df = df.sort_values(['date', 'tic'], ignore_index=True)

#Understand where to cut the dataset
colonna= df["tic"].tolist()
indici=[]
for el in lista:
    indici.append(colonna.index(el))
newdf= df.truncate(6378)
df= newdf

#########################################################################################################################
#Feature Engeneering
##########################################################################################################################

#Add Thecnical Indicators
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

#TRAINIG MODEL A2C

agent = DRLAgent(env = env_train)

A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.005, "learning_rate": 0.0002}
model_a2c = agent.get_model(model_name="a2c",model_kwargs = A2C_PARAMS)

trained_a2c = agent.train_model(model=model_a2c,
                                tb_log_name='a2c',
                                total_timesteps=60000)
trained_a2c.save(f"C:/Users/federico/Documents/Desktop/Tesi/trained_models/A2C_Crypto")

#TRAINING MODEL PPO
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

trained_ppo.save(f"C:/Users/federico/Documents/Desktop/Tesi/trained_models/PPOCrypto")


#TRAINING MODEL DDPG
agent_ddpg = DRLAgent(env = env_train)
DDPG_PARAMS = {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001}


model_ddpg = agent_ddpg.get_model("ddpg",model_kwargs = DDPG_PARAMS)

trained_ddpg = agent_ddpg.train_model(model=model_ddpg,
                             tb_log_name='ddpg',
                             total_timesteps=50000)
trained_ddpg.save(f"C:/Users/federico/Documents/Desktop/Tesi/trained_models/DDPGCrypto")

#TRAINING MODEL SAC

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
trained_sac.save(f"C:/Users/federico/Documents/Desktop/Tesi/trained_models/SACCrypto ")