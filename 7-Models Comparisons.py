import pandas as pd
import os
dic={1:"A2C",2:"DDPG",3:"PPO",4:"SAC"}

#For Stocks
i=1
for el in os.listdir("C:/Users/federico/Documents/Desktop/Tesi/results/stats"):
    if el[:-4].endswith("prova"):
        df=pd.read_csv("C:/Users/federico/Documents/Desktop/Tesi/results/stats"+"/"+str(el))
        df = df.rename(columns={"Unnamed: 0": "index"+str(i), "0": dic[i]})
        i+=1
        try:
            df_tot_prova=pd.concat([df_tot_prova,df], axis=1)
        except:
            df_tot_prova= pd.DataFrame(df)
df_tot_prova=df_tot_prova.drop(["index2","index3","index4"],axis=1)
df_tot_prova.rename(columns={"index1":"Market_Stats"},inplace=True)

#For Crypto
i=1
for el in os.listdir("C:/Users/federico/Documents/Desktop/Tesi/results/stats"):
    if el[:-4].endswith("crypto"):
        df=pd.read_csv("C:/Users/federico/Documents/Desktop/Tesi/results/stats"+"/"+str(el))
        df = df.rename(columns={"Unnamed: 0": "index"+str(i), "0": dic[i]})
        i+=1
        try:
            df_tot_crypto=pd.concat([df_tot_crypto,df], axis=1)
        except:
            df_tot_crypto= pd.DataFrame(df)
df_tot_crypto=df_tot_crypto.drop(["index2","index3","index4"],axis=1)
df_tot_crypto.rename(columns={"index1":"Crypto_Stats"},inplace=True)

#For Bonds
i=1
for el in os.listdir("C:/Users/federico/Documents/Desktop/Tesi/results/stats"):
    if el[:-4].endswith("bond"):
        df=pd.read_csv("C:/Users/federico/Documents/Desktop/Tesi/results/stats"+"/"+str(el))
        df = df.rename(columns={"Unnamed: 0": "index"+str(i), "0": dic[i]})
        i+=1
        try:
            df_tot_bond=pd.concat([df_tot_bond,df], axis=1)
        except:
            df_tot_bond= pd.DataFrame(df)
df_tot_bond=df_tot_bond.drop(["index2","index3","index4"],axis=1)
df_tot_bond.rename(columns={"index1":"Bond_Stats"},inplace=True)