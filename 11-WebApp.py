import streamlit as st
import pandas as pd
from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.config import config
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from finrl.env import env_stocktrading
import time
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
import time
import os
import pyfolio.timeseries
#######################################################################################################################################
#VARIABLE SETTING(for later use)
###########################################################################
Path=None
st.markdown("RL POWERED TRADING STRATEGY APP")
st.write("__________________________________________________________________")
st.write("The app will first evaluate the performance of the chosen portfolio as if the assets were equally distributed (Baseline).")
st.write("It will then compute a Reinforcement Learning Powered strategy to optimize the return of the portfolio by rebalancing the proportion of the assets!")
st.write("Finally it will provide some useful statistics to compare the two strategies.")
st.write("__________________________________________________________________")
st.write("<-<-<-Select the desired porfolio on the left <-<-<-")

Portfolio= []
port= "Your Portfolio is composed of: "
asset_2,asset_3,asset_4,asset_5,asset_6,asset_7,asset_8,asset_9,asset_10,asset_11,asset_12,asset_13,asset_14,asset_15="Select","Select","Select","Select","Select","Select","Select","Select","Select","Select","Select","Select","Select","Select"
ticker_list=[]
#Firms dictionary for matching user selection to tikers############################################################
firms={
        "Apple":"AAPL",
        "Microsoft":"MSFT",
        "JPMorgan":"JPM",
        "Visa":"V",
        "Raytheon Technologies":"RTX",
        "The Procter & Gamble Company":"PG",
        "Goldman Sachs Group":"GS",
        "Nike":"NKE",
        "Disney":"DIS",
        "American Express":"AXP",
        "Home Depot":"HD",
        "Intel":"INTC",
        "Walmart":"WMT",
        "IBM":"IBM",
        "Merck":"MRK",
        "United Health Group":"UNH",
        "Coca-Cola":"KO",
        "Caterpillar":"CAT",
        "The Travelers Companies":"TRV",
        "Johnson and Johnson":"JNJ",
        "Chevron Corporation":"CVX",
        "McDonald's":"MCD",
        "Verizon Communications":"VZ",
        "Cisco":"CSCO",
        "Exxon Mobil":"XOM",
        "Boeing":"BA",
        "3M Company":"MMM",
        "Pfizer":"PFE",
        "Walgreens Boots Alliance":"WBA",
        "DuPont":"DD",
}
firms_reverse={
        "AAPL":"Apple",
        "MSFT":"Microsoft",
        "JPM":"JPMorgan",
        "V":"Visa",
        "RTX":"Raytheon Technologies",
        "PG":"The Procter & Gamble Company",
        "GS":"Goldman Sachs Group",
        "NKE":"Nike",
        "DIS":"Disney",
        "AXP":"American Express",
        "HD":"Home Depot",
        "INTC":"Intel",
        "WMT":"Walmart",
        "IBM":"IBM",
        "MRK":"Merck",
        "UNH":"United Health Group",
        "KO":"Coca-Cola",
        "CAT":"Caterpillar",
        "TRV":"The Travelers Companies",
        "JNJ":"Johnson and Johnson",
        "CVX":"Chevron Corporation",
        "MCD":"McDonald's",
        "VZ":"Verizon Communications",
        "CSCO":"Cisco",
        "XOM":"Exxon Mobil",
        "BA":"Boeing",
        "MMM":"3M Company",
        "PFE":"Pfizer",
        "WBA":"Walgreens Boots Alliance",
        "DD":"DuPont",
}
#Constucting lists for the selection boxes.
#Since streamlit would rerun this piece of code each time a user interacts with a widget,
#it is commented here to reuce computations

#k=[]
#c=1
#while c<16:
    #holder=[]
    #for el in firms.keys():
        #holder.append(el)
    #holder[c],holder[-c]=holder[-c],holder[c]
    #k.append(holder)
    #c+=1

###########################################################################################################################################
#PORTFOLIO SELECTION
###########################################################################################################################################

asset_1= st.sidebar.selectbox(
    'Select your next asset',
    ("Select",'Apple', 'DuPont', 'JPMorgan', 'Visa', 'Raytheon Technologies', 'The Procter & Gamble Company', 'Goldman Sachs Group', 'Nike', 'Disney', 'American Express', 'Home Depot', 'Intel', 'Walmart', 'IBM', 'Merck', 'United Health Group', 'Coca-Cola', 'Caterpillar', 'The Travelers Companies', 'Johnson and Johnson', 'Chevron Corporation', "McDonald's", 'Verizon Communications', 'Cisco', 'Exxon Mobil', 'Boeing', '3M Company', 'Pfizer', 'Walgreens Boots Alliance', 'Microsoft')
)
if asset_1!= "Select":
    Portfolio.append(asset_1)
    asset_2 = st.sidebar.selectbox(
        'Select your next asset',
        ("Select",'Apple', 'Microsoft', 'Walgreens Boots Alliance', 'Visa', 'Raytheon Technologies', 'The Procter & Gamble Company', 'Goldman Sachs Group', 'Nike', 'Disney', 'American Express', 'Home Depot', 'Intel', 'Walmart', 'IBM', 'Merck', 'United Health Group', 'Coca-Cola', 'Caterpillar', 'The Travelers Companies', 'Johnson and Johnson', 'Chevron Corporation', "McDonald's", 'Verizon Communications', 'Cisco', 'Exxon Mobil', 'Boeing', '3M Company', 'Pfizer', 'JPMorgan', 'DuPont')
    )
if asset_2!= "Select":
    Portfolio.append(asset_2)
    asset_3 = st.sidebar.selectbox(
        'Select your next asset',
        ("Select",'Apple', 'Microsoft', 'JPMorgan', 'Pfizer', 'Raytheon Technologies', 'The Procter & Gamble Company', 'Goldman Sachs Group', 'Nike', 'Disney', 'American Express', 'Home Depot', 'Intel', 'Walmart', 'IBM', 'Merck', 'United Health Group', 'Coca-Cola', 'Caterpillar', 'The Travelers Companies', 'Johnson and Johnson', 'Chevron Corporation', "McDonald's", 'Verizon Communications', 'Cisco', 'Exxon Mobil', 'Boeing', '3M Company', 'Visa', 'Walgreens Boots Alliance', 'DuPont')
    )
if asset_3!= "Select":
    Portfolio.append(asset_3)
    asset_4 = st.sidebar.selectbox(
        'Select your next asset',
        ("Select",'Apple', 'Microsoft', 'JPMorgan', 'Visa', '3M Company', 'The Procter & Gamble Company', 'Goldman Sachs Group', 'Nike', 'Disney', 'American Express', 'Home Depot', 'Intel', 'Walmart', 'IBM', 'Merck', 'United Health Group', 'Coca-Cola', 'Caterpillar', 'The Travelers Companies', 'Johnson and Johnson', 'Chevron Corporation', "McDonald's", 'Verizon Communications', 'Cisco', 'Exxon Mobil', 'Boeing', 'Raytheon Technologies', 'Pfizer', 'Walgreens Boots Alliance', 'DuPont')
    )
if asset_4!= "Select":
    Portfolio.append(asset_4)
    asset_5 = st.sidebar.selectbox(
        'Select your next asset',
        ("Select",'Apple', 'Microsoft', 'JPMorgan', 'Visa', 'Raytheon Technologies', 'Boeing', 'Goldman Sachs Group', 'Nike', 'Disney', 'American Express', 'Home Depot', 'Intel', 'Walmart', 'IBM', 'Merck', 'United Health Group', 'Coca-Cola', 'Caterpillar', 'The Travelers Companies', 'Johnson and Johnson', 'Chevron Corporation', "McDonald's", 'Verizon Communications', 'Cisco', 'Exxon Mobil', 'The Procter & Gamble Company', '3M Company', 'Pfizer', 'Walgreens Boots Alliance', 'DuPont')
    )

if asset_5!= "Select":
    Portfolio.append(asset_5)
    asset_6 = st.sidebar.selectbox(
        'Select your next asset',
        ("Select",'Apple', 'Microsoft', 'JPMorgan', 'Visa', 'Raytheon Technologies', 'The Procter & Gamble Company', 'Exxon Mobil', 'Nike', 'Disney', 'American Express', 'Home Depot', 'Intel', 'Walmart', 'IBM', 'Merck', 'United Health Group', 'Coca-Cola', 'Caterpillar', 'The Travelers Companies', 'Johnson and Johnson', 'Chevron Corporation', "McDonald's", 'Verizon Communications', 'Cisco', 'Goldman Sachs Group', 'Boeing', '3M Company', 'Pfizer', 'Walgreens Boots Alliance', 'DuPont')
    )

if asset_6!= "Select":
    Portfolio.append(asset_6)
    asset_7 = st.sidebar.selectbox(
        'Select your next asset',
        ("Select",'Apple', 'Microsoft', 'JPMorgan', 'Visa', 'Raytheon Technologies', 'The Procter & Gamble Company', 'Goldman Sachs Group', 'Cisco', 'Disney', 'American Express', 'Home Depot', 'Intel', 'Walmart', 'IBM', 'Merck', 'United Health Group', 'Coca-Cola', 'Caterpillar', 'The Travelers Companies', 'Johnson and Johnson', 'Chevron Corporation', "McDonald's", 'Verizon Communications', 'Nike', 'Exxon Mobil', 'Boeing', '3M Company', 'Pfizer', 'Walgreens Boots Alliance', 'DuPont')
    )

if asset_7!= "Select":
    Portfolio.append(asset_7)
    asset_8 = st.sidebar.selectbox(
        'Select your next asset',
        ("Select",'Apple', 'Microsoft', 'JPMorgan', 'Visa', 'Raytheon Technologies', 'The Procter & Gamble Company', 'Goldman Sachs Group', 'Nike', 'Verizon Communications', 'American Express', 'Home Depot', 'Intel', 'Walmart', 'IBM', 'Merck', 'United Health Group', 'Coca-Cola', 'Caterpillar', 'The Travelers Companies', 'Johnson and Johnson', 'Chevron Corporation', "McDonald's", 'Disney', 'Cisco', 'Exxon Mobil', 'Boeing', '3M Company', 'Pfizer', 'Walgreens Boots Alliance', 'DuPont')
)

if asset_8!= "Select":
    Portfolio.append(asset_8)
    asset_9 = st.sidebar.selectbox(
        'Select your next asset',
        ("Select",'Apple', 'Microsoft', 'JPMorgan', 'Visa', 'Raytheon Technologies', 'The Procter & Gamble Company', 'Goldman Sachs Group', 'Nike', 'Disney', "McDonald's", 'Home Depot', 'Intel', 'Walmart', 'IBM', 'Merck', 'United Health Group', 'Coca-Cola', 'Caterpillar', 'The Travelers Companies', 'Johnson and Johnson', 'Chevron Corporation', 'American Express', 'Verizon Communications', 'Cisco', 'Exxon Mobil', 'Boeing', '3M Company', 'Pfizer', 'Walgreens Boots Alliance', 'DuPont'
         )
    )
if asset_9!= "Select":
    Portfolio.append(asset_9)
    asset_10 = st.sidebar.selectbox(
        'Select your next asset',
        ("Select",'Apple', 'Microsoft', 'JPMorgan', 'Visa', 'Raytheon Technologies', 'The Procter & Gamble Company', 'Goldman Sachs Group', 'Nike', 'Disney', 'American Express', 'Chevron Corporation', 'Intel', 'Walmart', 'IBM', 'Merck', 'United Health Group', 'Coca-Cola', 'Caterpillar', 'The Travelers Companies', 'Johnson and Johnson', 'Home Depot', "McDonald's", 'Verizon Communications', 'Cisco', 'Exxon Mobil', 'Boeing', '3M Company', 'Pfizer', 'Walgreens Boots Alliance', 'DuPont'
         )
    )

if asset_10!= "Select":
    Portfolio.append(asset_10)
    asset_11 = st.sidebar.selectbox(
        'Select your next asset',
        ("Select",'Apple', 'Microsoft', 'JPMorgan', 'Visa', 'Raytheon Technologies', 'The Procter & Gamble Company', 'Goldman Sachs Group', 'Nike', 'Disney', 'American Express', 'Home Depot', 'Johnson and Johnson', 'Walmart', 'IBM', 'Merck', 'United Health Group', 'Coca-Cola', 'Caterpillar', 'The Travelers Companies', 'Intel', 'Chevron Corporation', "McDonald's", 'Verizon Communications', 'Cisco', 'Exxon Mobil', 'Boeing', '3M Company', 'Pfizer', 'Walgreens Boots Alliance', 'DuPont'
         )
    )

if asset_11!= "Select":
    Portfolio.append(asset_11)
    asset_12 = st.sidebar.selectbox(
        'Select your next asset',
        ("Select",'Apple', 'Microsoft', 'JPMorgan', 'Visa', 'Raytheon Technologies', 'The Procter & Gamble Company', 'Goldman Sachs Group', 'Nike', 'Disney', 'American Express', 'Home Depot', 'Intel', 'The Travelers Companies', 'IBM', 'Merck', 'United Health Group', 'Coca-Cola', 'Caterpillar', 'Walmart', 'Johnson and Johnson', 'Chevron Corporation', "McDonald's", 'Verizon Communications', 'Cisco', 'Exxon Mobil', 'Boeing', '3M Company', 'Pfizer', 'Walgreens Boots Alliance', 'DuPont')
         )

if asset_12!= "Select":
    Portfolio.append(asset_12)
    asset_13 = st.sidebar.selectbox(
        'Select your next asset',
        ("Select",'Apple', 'Microsoft', 'JPMorgan', 'Visa', 'Raytheon Technologies', 'The Procter & Gamble Company', 'Goldman Sachs Group', 'Nike', 'Disney', 'American Express', 'Home Depot', 'Intel', 'Walmart', 'Caterpillar', 'Merck', 'United Health Group', 'Coca-Cola', 'IBM', 'The Travelers Companies', 'Johnson and Johnson', 'Chevron Corporation', "McDonald's", 'Verizon Communications', 'Cisco', 'Exxon Mobil', 'Boeing', '3M Company', 'Pfizer', 'Walgreens Boots Alliance', 'DuPont'
         )
    )

if asset_13!= "Select":
    Portfolio.append(asset_13)
    asset_14 = st.sidebar.selectbox(
        'Select your next asset',
        ("Select",'Apple', 'Microsoft', 'JPMorgan', 'Visa', 'Raytheon Technologies', 'The Procter & Gamble Company', 'Goldman Sachs Group', 'Nike', 'Disney', 'American Express', 'Home Depot', 'Intel', 'Walmart', 'IBM', 'Coca-Cola', 'United Health Group', 'Merck', 'Caterpillar', 'The Travelers Companies', 'Johnson and Johnson', 'Chevron Corporation', "McDonald's", 'Verizon Communications', 'Cisco', 'Exxon Mobil', 'Boeing', '3M Company', 'Pfizer', 'Walgreens Boots Alliance', 'DuPont'
         )
    )

if asset_14!= "Select":
    Portfolio.append(asset_14)
    asset_15 = st.sidebar.selectbox(
        'Select your next asset',
        ("Select",'Apple', 'Microsoft', 'JPMorgan', 'Visa', 'Raytheon Technologies', 'The Procter & Gamble Company', 'Goldman Sachs Group', 'Nike', 'Disney', 'American Express', 'Home Depot', 'Intel', 'Walmart', 'IBM', 'Merck', 'United Health Group', 'Coca-Cola', 'Caterpillar', 'The Travelers Companies', 'Johnson and Johnson', 'Chevron Corporation', "McDonald's", 'Verizon Communications', 'Cisco', 'Exxon Mobil', 'Boeing', '3M Company', 'Pfizer', 'Walgreens Boots Alliance', 'DuPont'
         )
    )

if asset_15!= "Select":
    Portfolio.append(asset_15)


for el in Portfolio:
    if len(Portfolio)==1:
        port = str(port)+ " " + str(el)
    else:
        port= str(port)+","+" "+str(el)
if len(set(Portfolio))!=len(Portfolio):
    st.write("ALLERT:")
    st.write("New Selected Asset Already Present In The Portfolio")
    st.write("For a better evaluation of the strategy please select a different asset")
else:
    st.write(port)
    st.write(str(len(Portfolio))+"/15")
st.write("__________________________________________________________________")
Path = st.text_input("If you want to download the model for later use just type the folder path! This is completely optional!")
start=st.button("START EVALUATING STRATEGY")
st.write("__________________________________________________________________")
#######################################################################################################################
# BASELINE RETRIEVING AND STATS EVALUATION
######################################################################################################################
if len(Portfolio)==15 or start==True:
    st.write("Starting Baseline Evaluation")
    time.sleep(2)
    st.write("Retrieving Data...")
    for el in Portfolio:
        ticker_list.append(firms[el])

    df = YahooDownloader(start_date='2009-01-01',
                         end_date='2021-06-01',
                         ticker_list=ticker_list).fetch_data()
    st.write("Data Retrieving Complete!")
    st.write("__________________________________________________________________")
    time.sleep(2)
    st.write("Preprocessing Data...")
    #Add Financial Indicators for the model
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
    st.write("Evaluating Baseline Portfolio Performances...")

    # Evaluating daily portfolio returns
    df["Date Index"] = df.date.factorize()[0]
    returns_list = [[float("nan"), "2009-12-30"]]
    for data in df["Date Index"].unique().tolist()[0:-1]:
        day = df[df["Date Index"] == data]
        for el in day.index:
            daily_ret = []
            close1 = day.at[el, "close"]
            next = el + len(ticker_list)
            close2 = df.at[next, "close"]
            returns = close2 / close1 - 1
            daily_ret.append(returns)
        value = 0
        for ret in daily_ret:
            value += ret / len(daily_ret)
        returns_list.append([value, day.at[el, "date"]])

    # converting the results into a df

    date = []
    for el in returns_list:
        date.append(el[1])
    returns_series = pd.DataFrame(returns_list, index=date, columns=["daily_return", "date"])
    DRL_strat = backtest.convert_daily_return_to_pyfolio_ts(returns_series)

    # Evaluating strategy performances

    perf_func = pyfolio.timeseries.perf_stats

    st.write("__________________________________________________________________")
    #180 trading days perf
    st.write("Baseline Portfolio Performances In The Last 180 Trading Days")
    series_180 = DRL_strat[-180:]
    dat_180=pd.DataFrame(DRL_strat[-180:], columns=["Cumulative Returns Baseline 180"])
    st.line_chart(dat_180)
    perf_stats_all_180 = perf_func(returns=series_180,
                                   factor_returns=series_180,
                                   positions=None, transactions=None, turnover_denom="AGB")
    perf_stats_all_180=perf_stats_all_180[1:-2]
    st.write(perf_stats_all_180)
    st.write("__________________________________________________________________")
    time.sleep(5)

    # 360 trading days perf
    st.write("Baseline Portfolio Performances In The Last 360 Trading Days")
    series_360 = DRL_strat[-360:]
    dat_360 = pd.DataFrame(DRL_strat[-360:], columns=["Cumulative Returns Baseline 360"])
    st.line_chart(dat_360)
    perf_stats_all_360 = perf_func(returns=series_360,
                                  factor_returns=series_360,
                                  positions=None, transactions=None, turnover_denom="AGB")
    perf_stats_all_360 = perf_stats_all_360[1:-2]
    st.write(perf_stats_all_360)
    st.write("__________________________________________________________________")
    time.sleep(8)

    #start training the model
    st.write("Starting To Evaluate a Customized Strategy")
    time.sleep(2)
    st.write("Setting up the Envirorment...")

    #Env Setup

    train = data_split(df, '2009-01-01', '2019-01-01')
    trade = data_split(df, '2019-01-01', '2021-06-01')
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

    st.write("Env Setup Succesfully Executed!")
    st.write("__________________________________________________________________")
    time.sleep(2)
    st.write("Start Training the Agent...")

    #Training PPO Agent adopting Grid seacrh Params

    agent_ppo = DRLAgent(env=env_train)
    PPO_PARAMS = {
        "n_steps": 1024,
        "ent_coef": 0.005,
        "learning_rate": 0.0001,
        "batch_size": 256,
    }
    model_ppo = agent_ppo.get_model("ppo", model_kwargs=PPO_PARAMS)

    trained_ppo = agent_ppo.train_model(model=model_ppo,
                                        tb_log_name='ppo',
                                        total_timesteps=80000)
    st.write("Agent Succesfully trained")
    time.sleep(2)
    st.write("Evaluating Agent Strategy Performances...")
    time.sleep(5)

    #Predictions and Performances Evaluation

    df_daily_return, df_actions = DRLAgent.DRL_prediction(model=trained_ppo, environment=e_trade_gym)

    DRL_strat_agent = backtest.convert_daily_return_to_pyfolio_ts(df_daily_return)

    perf_stats_all_agent = perf_func(returns=DRL_strat_agent,
                               factor_returns=DRL_strat_agent,
                               positions=None, transactions=None, turnover_denom="AGB")
    #Presenting Agent Performances

    st.write("__________________________________________________________________")
    # 180 trading days perf
    st.write("Agent Strategy Portfolio Performances In The Last 180 Trading Days")
    series_180_agent = DRL_strat_agent[-180:]
    dat_180_agent = pd.DataFrame(DRL_strat_agent[-180:], columns=["Cumulative Returns Agent Strategy 180 "])
    st.line_chart(dat_180_agent)
    perf_stats_all_180_agent = perf_func(returns=series_180_agent,
                                   factor_returns=series_180_agent,
                                   positions=None, transactions=None, turnover_denom="AGB")
    perf_stats_all_180_agent = perf_stats_all_180_agent[1:-2]
    st.write(perf_stats_all_180_agent)
    st.write("__________________________________________________________________")
    time.sleep(5)

    # 360 trading days perf
    st.write("Agent Strategy Portfolio Performances In The Last 360 Trading Days")
    series_360_agent = DRL_strat_agent[-360:]
    dat_360_agent = pd.DataFrame(DRL_strat_agent[-360:], columns=["Cumulative Returns Agent Strategy 360 "])
    st.line_chart(dat_360_agent)
    perf_stats_all_360_agent = perf_func(returns=series_360_agent,
                                        factor_returns=series_360_agent,
                                        positions=None, transactions=None, turnover_denom="AGB")
    perf_stats_all_360_agent = perf_stats_all_360_agent[1:-2]
    st.write(perf_stats_all_360_agent)
    st.write("__________________________________________________________________")
    time.sleep(5)

    #Compared Performances on 180 days
    st.write("Compared Performances over 180 Trading Days")
    compared_returns_180= pd.concat([dat_180_agent,dat_180], axis=1)
    st.line_chart(compared_returns_180)
    st.write("__________________________________________________________________")
    # Compared Performances on 360 days
    st.write("Compared Performances over 360 Trading Days ")
    compared_returns_360 = pd.concat([dat_360_agent, dat_360], axis=1)
    st.line_chart(compared_returns_360)
    st.write("__________________________________________________________________")

    #Present Convlusions
    diff_360= perf_stats_all_360_agent["Cumulative returns"]-perf_stats_all_360["Cumulative returns"]
    diff_180= perf_stats_all_180_agent["Cumulative returns"]-perf_stats_all_180["Cumulative returns"]
    if diff_180 > 0:
        st.write("Agent Strategy Porfolio achieved returns of " + str(abs(diff_180))[2:4]+"."+ str(abs(diff_180))[5]+"%"+" higher than the Baseline on a 180 trading days period")
    elif diff_180 < 0:
        st.write("Agent Strategy Porfolio achieved returns of " + str(abs(diff_180))[2:4] +"."+ str(abs(diff_180))[5]+"%"+ "smaller than the Baseline on a 180 trading days period")
    if diff_360>0:
        st.write("Agent Strategy Porfolio achieved returns of " + str(abs(diff_360))[2:4] +"."+ str(abs(diff_180))[5]+"%"+ " higher than the Baseline on a 360 trading days period")
    elif diff_360<0:
        st.write("Agent Strategy Porfolio achieved returns of " + str(abs(diff_360))[2:4] +"."+ str(abs(diff_180))[5]+"%"+ " smaller than the Baseline on a 360 trading days period")

    #AVG and SD of Portfolio composition
    st.write("__________________________________________________________________")
    st.write("These are the Average and Standard Deviation of the proportions of the Agent Strategy portfolio!")
    useful_dic_180={}
    useful_dic_360={}
    st.write("For 180 Trading Days")
    for el in df_actions.columns:
        useful_dic_180[el]=[df_actions[el][-180:].mean(),df_actions[el][-180:].std()]
    for el in df_actions.columns:
        useful_dic_360[el]=[df_actions[el][-360:].mean(),df_actions[el][-360:].std()]
    useful_columns={}
    for el in df_actions.columns:
        useful_columns[el]=firms_reverse[el]
    useful_data_180=pd.DataFrame(useful_dic_180, index=["Mean","Standard Deviation"])
    useful_data_180.rename(columns=useful_columns,inplace=True)
    useful_data_360 = pd.DataFrame(useful_dic_360, index=["Mean", "Standard Deviation"])
    useful_data_360.rename(columns=useful_columns, inplace=True)
    st.write(useful_data_180)
    time.sleep(4)
    st.write("__________________________________________________________________")
    st.write("For 360 Trading Days")
    st.write(useful_data_360)
    time.sleep(5)
    st.write("__________________________________________________________________")
    if Path!=None:
        st.write("Saving the strategy at "+"'"+ str(Path)+"\RL_Portfolio_Strategy"+"'"+"...")
        trained_ppo.save(str(Path)+ "\RL_Portfolio_Strategy")
        st.write("__________________________________________________________________")
    st.write("Thanks for trying this app!")


