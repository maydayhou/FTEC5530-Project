
# coding: utf-8

# In[1]:


import datetime
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="whitegrid")
from scipy.spatial.distance import pdist
from numpy import cumsum, log, polyfit, sqrt, std, subtract
from numpy.random import randn


# In[2]:


def get_stockData(stock_list,s_y,s_m,s_d,e_y,e_m,e_d):
    d = pd.DataFrame()
    start = datetime.datetime(s_y,s_m,s_d) # or start = '1/1/2016'
    end = datetime.datetime(e_y,e_m,e_d)
    k = 0 
    for i in stock_list:
            prices = web.DataReader(i,'yahoo', start, end)
            d1 = prices['Adj Close']
            d = pd.concat([d, d1], axis=1)
    d.columns = stock_list
    return(d)


# In[3]:


def get_distance(x):
    df = pd.DataFrame(data=[[0,0,0]])
    list1 = x.columns
    for i in range(0,19):
        for j in range(i+1,20):
            distance = pdist(x.iloc[:, [i,j]].T, 'euclidean')[0]
            my_df = pd.DataFrame(data=[[list1[i],list1[j],distance]])
            df = pd.concat([df, my_df], axis=0)
    df = df.reset_index(drop=True)
    df = df.drop([0])
    df.columns = ["Stock1","Stock2","Distance"]
    df = df.sort_values(by =['Distance'],ascending=True)

    return(df)


# In[4]:


def get_cointergrated(data):
    n = data.shape[1]
    df = pd.DataFrame(data=[[0,0,0]])
    keys = data.keys()
    for i in range(n):
        for j in range(i+1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            result = coint(S1, S2)
            my_df = pd.DataFrame(data=[[keys[i],keys[j],result[0],result[1]]])
            df = pd.concat([df,my_df], axis=0)
            
    df = df.reset_index(drop=True)
    df = df.drop([0])
    df.columns = ["Stock1","Stock2","score","p-value"]

    df = df.sort_values(by =['p-value'],ascending=True)

    return(df)


# In[5]:


def hurst(ts):
    lags = range(2, 100)
    tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    poly = polyfit(log(lags), log(tau), 1)
    return poly[0]*2.0


# In[6]:


def zscore(x):
    return (x - x.mean()) / np.std(x)


# In[7]:


def  cal_spread(S1,S2):
    S1 = sm.add_constant(S1)
    results = sm.OLS(S2, S1).fit()
    b = results.params[1]
    S1 = S1.iloc[:,1] 
    spread = S2 - b * S1
    return(spread,b)


# In[8]:


def cal_beta(S1,S2):
    S1 = sm.add_constant(S1)
    results = sm.OLS(S2, S1).fit()
    b = results.params[1]
    return(b)


# In[9]:


def get_hurst(stock_data):
    
    n = stock_data.shape[1]   
    keys = stock_data.keys()
    df = pd.DataFrame(data=[[0,0,0]])
    
    for i in range(n):
        for j in range(i+1, n):
            x = stock_data[keys[i]]
            y = stock_data[keys[j]]                  
            series = x - y
            h = hurst(series)
            my_df = pd.DataFrame(data=[[keys[i],keys[j],h]])
            df = pd.concat([df,my_df], axis=0)
    df = df.reset_index(drop=True)
    df = df.drop([0])
    df.columns = ["Stock1","Stock2","Hurst"]
    df = df.sort_values(by =['Hurst'],ascending=True)

    return (df)


# In[10]:


def get_plot0(x,y,z):

    plt.figure(figsize=(12, 6))
    plt.plot(x.index, x.values)
    plt.plot(y.index, y.values)
    plt.plot(z.index, z.values)
    plt.legend(['spread/distance', '5d  MA', '60d  MA'])

    plt.ylabel('spread/distance')
    plt.show()


# In[11]:


def get_plot1(x,y):
   plt.figure(figsize=(12,6))
   x.plot()
   plt.axhline(0, color='black')
   plt.axhline(y ,color='red', linestyle='--')
   plt.axhline(-y, color='green', linestyle='--')
   plt.legend(['Rolling Spread z-Score', 'Mean', 'up','down'])
   plt.show()


# In[12]:


def get_plot2(x,y,z):
    y = int(y)
    plt.figure(figsize=(12,6))
    x.plot()
    buy = x.copy()
    sell = x.copy()
    buy[z> (-y)] = 0
    sell[z<y] = 0
    buy[lambda x: x!=0].plot(color='g', linestyle='None', marker='^')
    sell[lambda x: x!=0].plot(color='r', linestyle='None', marker='^')
    x1, x2, y1, y2 = plt.axis()
    plt.legend(['Spread/distance', 'Buy Signal', 'Sell Signal'])
    plt.show()
    return(buy,sell)


# In[13]:


def get_plot3(S1,S2,buy,sell):
  plt.figure(figsize=(12,7))
  S1[59:].plot(color='b')
  S2[59:].plot(color='c')
  buyR = 0*S1.copy()
  sellR = 0*S1.copy()

  buyR[buy!=0] = S1[buy!=0]
  sellR[buy!=0] = S2[buy!=0]

  buyR[sell!=0] = S2[sell!=0]
  sellR[sell!=0] = S1[sell!=0]

  buyR[lambda x: x!=0].plot(color='g', linestyle='None', marker='^')
  sellR[lambda x: x!=0].plot(color='r', linestyle='None', marker='^')
  x1, x2, y1, y2 = plt.axis()
  plt.axis((x1, x2, min(S1.min(), S2.min()), max(S1.max(), S2.max())))
  plt.xlim('2018-01-01', '2019-12-31')

  plt.legend(['S1', 'S2', 'Buy Signal', 'Sell Signal'])
  plt.show()


# In[14]:


def coin_train(x,y,k,t):
    k = int(k)
    list1 = [x,y]
    stock_data = get_stockData(list1,2017,1,1,2019,12,31)
    stock_data = stock_data.dropna()
    stock_data.columns = list1
    pd.concat([stock_data.iloc[:,0],stock_data.iloc[:,1]], axis=1).plot(figsize=(12,6))
    plt.show()
    beta = cal_beta(stock_data[stock_data.index<'2018-01-01'].iloc[:,0],
                    stock_data[stock_data.index<'2018-01-01'].iloc[:,1])
    stock_data['spread'] = stock_data.iloc[:,1]-beta*stock_data.iloc[:,0]
    spread = stock_data[stock_data.index>='2018-01-01'].iloc[:,2]
    ## splite the train set and test set
    S1 = stock_data[stock_data.index>='2018-01-01'].iloc[:,0]
    S2 = stock_data[stock_data.index>='2018-01-01'].iloc[:,1]
    S1_1 = stock_data.iloc[:,0]
    S2_1 = stock_data.iloc[:,0]

    ma1 = spread.rolling(window=5, center=False).mean()
    ma2 = spread.rolling(window=60, center=False).mean()
    std = spread.rolling(window=60, center=False).std()
    zscore = (ma1 - ma2)/std
    money = 0
    countS1 = 0
    countS2 = 0
    
    get_plot0(spread,ma1,ma2)
    get_plot1(zscore,k)
    buy,sell = get_plot2(spread,k,zscore)
    get_plot3(S1,S2,buy,sell)

    for i in range(len(spread)):
        beta = cal_beta(S1_1[:S1.index[i]],S2_1[:S2.index[i]])
        if zscore[i] < -k:
            money += S1[i]* beta - S2[i]
            fee = 2*(S1[i]* beta+S2[i])*0.003
            money -= fee
            countS1 -= beta
            countS2 += 1

        elif zscore[i] > k:
            money -= S1[i]*beta - S2[i] 
            fee = 2*(S1[i]* beta+S2[i])*0.003
            money -= fee
            countS1 += beta
            countS2 -= 1
# 
        elif abs(zscore[i]) == t:
            money += S1[i] * countS1 + S2[i] * countS2
            fee = 2*(abs(S1[i] * countS1)+abs(S2[i] * countS2))*0.003
            money -= fee
            countS1 = 0
            countS2 = 0            
    return(money)

s_list = ['1579.HK','0168.HK','0291.HK','1112.HK','2319.HK','0345.HK','1886.HK','1583.HK','1068.HK','0322.HK',
          '1717.HK','0220.HK','1117.HK','0151.HK','3799.HK','1533.HK','6868.HK','2218.HK','1458.HK','0374.HK']
s_data = get_stockData(s_list,2017,4,1,2017,12,31)
s_data1 = s_data.dropna(axis=0, how='any', inplace=False)
data_norm = s_data1.apply(lambda x: (x-np.mean(x))/np.std(x))


# In[15]:


def coin_test(x,y,k,t):
    list1 = [x,y]
    stock_data = get_stockData(list1,2017,1,1,2017,12,31)
    stock_data = stock_data.dropna()
    stock_data.columns = list1
    stock_data['spread'],beta = cal_spread(stock_data.iloc[:,0],stock_data.iloc[:,1])
    spread = stock_data.iloc[:,2]
    S1 = stock_data.iloc[:,0]
    S2 = stock_data.iloc[:,0]

    ma1 = spread.rolling(window=5, center=False).mean()
    ma2 = spread.rolling(window=60, center=False).mean()
    std = spread.rolling(window=60, center=False).std()
    zscore = (ma1 - ma2)/std
    money = 0
    countS1 = 0
    countS2 = 0
    
    get_plot0(spread,ma1,ma2)
    get_plot1(zscore,k)

    for i in range(len(spread)):
        if zscore[i] < -k:
            money += S1[i]* beta - S2[i]
            fee = 2*(S1[i]* beta+S2[i])*0.003
            money -= fee
            countS1 -= beta
            countS2 += 1

        elif zscore[i] > k:
            money -= S1[i]*beta - S2[i] 
            fee = 2*(S1[i]* beta+S2[i])*0.003
            money -= fee
            countS1 += beta
            countS2 -= 1
# 
        elif abs(zscore[i]) == t:
            money += S1[i] * countS1 + S2[i] * countS2
            fee = 2*(abs(S1[i] * countS1)+abs(S2[i] * countS2))*0.003
            money -= fee
            countS1 = 0
            countS2 = 0            
    return(money)


# In[16]:


def select_para_coin(x):
    for i in range(len(x)):
        k_list = [1,1.5,2]
        for k in k_list:
            m1 = coin_test(x[i][0],x[i][1],k,0)
            m2 = coin_train(x[i][0],x[i][1],k,0)
            print(k,x[i][0],x[i][1],m1,m2)


# In[19]:


def distance_train (x,y,k,t):
    list1 = [x,y]
    stock_data = get_stockData(list1,2017,4,1,2019,12,31)
    stock_data = stock_data.dropna()
    stock_data.columns = list1
    
    stock_data_n = stock_data.apply(lambda x: (x-np.mean(x))/np.std(x))
    stock_data_n['distance'] = stock_data_n.iloc[:,0]-stock_data_n.iloc[:,1]
    distance = stock_data_n[stock_data_n.index>='2018-01-01'].iloc[:,2]
    
    pd.concat([stock_data_n.iloc[:,0],stock_data_n.iloc[:,1]], axis=1).plot(figsize=(12,6))
    plt.show()
    
    d1 = stock_data_n[stock_data_n.index>='2018-01-01'].iloc[:,0]
    d2 = stock_data_n[stock_data_n.index>='2018-01-01'].iloc[:,1]

    ma1 = distance.rolling(window=5, center=False).mean()
    ma2 = distance.rolling(window=60, center=False).mean()
    std = distance.rolling(window=60, center=False).std()
    zscore = (ma1 - ma2)/std
    money = 0
    countS1 = 0
    countS2 = 0
    
    get_plot1(zscore,k)
    buy,sell = get_plot2(distance,k,zscore)
    
    for i in range(len(distance)):
        if zscore[i] < -k:
            money -= d1[i] - d2[i]
            fee = 2*(d1[i]+d2[i])*0.003
            money -= fee
            countS1 += 1
            countS2 -= 1
        elif zscore[i] > k:
            money += d1[i] - d2[i] 
            fee = 2*(d1[i]+d2[i])*0.003
            money -= fee
            countS1 -= 1
            countS2 += 1

        elif abs(zscore[i]) == t:
            money += d1[i] * countS1 + d2[i] * countS2
            fee = 2*(abs(d1[i] * countS1)+abs(d2[i] * countS2))*0.003
            money -= fee
            countS1 = 0
            countS2 = 0
    return(money)


# In[20]:


def distance_test (x,y,k,t):
    list1 = [x,y]
    stock_data = get_stockData(list1,2017,4,1,2017,12,31)
    stock_data = stock_data.dropna()
    stock_data.columns = list1

    stock_data_n = stock_data.apply(lambda x: (x-np.mean(x))/np.std(x))
    stock_data_n['distance'] = stock_data_n.iloc[:,0]-stock_data_n.iloc[:,1]
    distance = stock_data_n.iloc[:,2]

    d1 = stock_data_n.iloc[:,0]
    d2 = stock_data_n.iloc[:,1]

    ma1 = distance.rolling(window=5, center=False).mean()
    ma2 = distance.rolling(window=60, center=False).mean()
    std = distance.rolling(window=60, center=False).std()
    zscore = (ma1 - ma2)/std
    money = 0
    countS1 = 0
    countS2 = 0
    
    get_plot1(zscore,k)
    
    for i in range(len(distance)):
        if zscore[i] < -k:
            money -= d1[i] - d2[i]
            fee = 2*(d1[i]+d2[i])*0.003
            money -= fee
            countS1 += 1
            countS2 -= 1

        elif zscore[i] > k:
            money += d1[i] - d2[i] 
            fee = 2*(d1[i]+d2[i])*0.003
            money -= fee
            countS1 -= 1
            countS2 += 1

        elif abs(zscore[i]) == t:
            money += d1[i] * countS1 + d2[i] * countS2
            fee = 2*(abs(d1[i] * countS1)+abs(d2[i] * countS2))*0.003
            money -= fee
            countS1 = 0
            countS2 = 0
            
    return(money)


# In[21]:


def select_para_dist(x):
    for i in range(len(x)):
        k_list = [1,1.5,2]
        for k in k_list:
            m1 = distance_test(x[i][0],x[i][1],k,0)
            m2 = distance_train(x[i][0],x[i][1],k,0)
            print(k,x[i][0],x[i][1],m1,m2)


# In[22]:


s_list = ['1579.HK','0168.HK','0291.HK','1112.HK','2319.HK','0345.HK','1886.HK','1583.HK','1068.HK','0322.HK',
          '1717.HK','0220.HK','1117.HK','0151.HK','3799.HK','1533.HK','6868.HK','2218.HK','1458.HK','0374.HK']
s_data = get_stockData(s_list,2017,4,1,2017,12,31)
s_data1 = s_data.dropna(axis=0, how='any', inplace=False)
data_norm = s_data1.apply(lambda x: (x-np.mean(x))/np.std(x))


# In[23]:


distance = get_distance(data_norm)
distance = distance.reset_index(drop=True)
distance.head(10)


# In[24]:


cointegation_result = get_cointergrated(s_data1)
cointegation_result = cointegation_result.reset_index(drop=True)
pairs_coin = cointegation_result[cointegation_result['p-value']<=0.05]
pairs_coin


# In[25]:


hurst_data = get_hurst(s_data1)
hurst_data[hurst_data['Hurst']<0.5].head(10)


# In[27]:


import seaborn as sns; sns.set(style="whitegrid")
distance_train ('2319.HK','3799.HK',1,0)


# In[28]:


distance_test ('2319.HK','3799.HK',1,0)


# In[29]:


distance_test ('1112.HK','3799.HK',1,0)


# In[30]:


distance_train ('1112.HK','3799.HK',1,0)


# In[31]:


distance_test ('1112.HK','0312.HK',1,0)


# In[32]:


distance_train ('1112.HK','0312.HK',1,0)


# In[36]:


coin_test('1533.HK',"1717.HK",2,0)


# In[37]:


coin_train('1533.HK',"1717.HK",2,0)


# In[39]:


coin_test('0345.HK',"0291.HK",2,0)


# In[41]:


coin_train('0345.HK',"0291.HK",2,0)


# In[42]:


coin_train('0345.HK',"6868.HK",2,0)


# In[43]:


coin_test('0345.HK',"6868.HK",2,0)


# In[44]:


coin_test('1068.HK',"0374.HK",1.5,0)


# In[45]:


coin_train('1068.HK',"0374.HK",1.5,0)

