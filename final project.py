# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 21:12:01 2023

@author: Shane
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import empyrical

data = pd.read_excel("model.xlsx")
data = data.dropna(axis = 0)
data.index=pd.date_range(start='1/29/1960',end='12/31/2010',freq='m')
del data['date']

# Forward rate

rx2=data['h(2,1)']-data['y1']
rx3=data['h(3,2)']-data['y1']
rx4=data['h(4,3)']-data['y1']
rx5=data['h(5,4)']-data['y1']

x = data[['y1','f(2,1)','f(3,2)','f(4,3)','f(5,4)']]
y = rx2
est2 = sm.OLS(y, sm.add_constant(x)).fit()
print(est2.summary())

x = data[['y1','f(2,1)','f(3,2)','f(4,3)','f(5,4)']]
y = rx3
est3 = sm.OLS(y, sm.add_constant(x)).fit()
print(est3.summary())

x = data[['y1','f(2,1)','f(3,2)','f(4,3)','f(5,4)']]
y = rx4
est4 = sm.OLS(y, sm.add_constant(x)).fit()
print(est4.summary())

x = data[['y1','f(2,1)','f(3,2)','f(4,3)','f(5,4)']]
y = rx5
est5 = sm.OLS(y, sm.add_constant(x)).fit()
print(est5.summary())


df=pd.DataFrame(columns=['2','3','4','5'])
df['2']=est2.params
y=rx3
est3 = sm.OLS(y, sm.add_constant(x)).fit()
df['3']=est3.params
y=rx4
est4 = sm.OLS(y, sm.add_constant(x)).fit()
df['4']=est4.params
y=rx5
est5 = sm.OLS(y, sm.add_constant(x)).fit()
df['5']=est5.params
df=df.drop(['const'])
plt.figure(figsize = (16,8))
df.plot()

rx_t=(rx2+rx3+rx4+rx5)/4
y=rx_t
x = data[['y1','f(2,1)','f(3,2)','f(4,3)','f(5,4)']]
est_single = sm.OLS(y, sm.add_constant(x)).fit()
print(est_single.summary())

x['1'] = 1
x1=est_single.params[0]*x['1']+est_single.params[1]*x['y1']+est_single.params[2]*x['f(2,1)']
+est_single.params[3]*x['f(3,2)']+est_single.params[4]*x['f(4,3)']+est_single.params[5]*x['f(5,4)']
y=rx2
est_restricted=sm.OLS(y, sm.add_constant(x1)).fit()
print(est_restricted.summary())

y=rx3
est_restricted3=sm.OLS(y, sm.add_constant(x1)).fit()
print(est_restricted3.summary())

y=rx4
est_restricted4=sm.OLS(y, sm.add_constant(x1)).fit()
print(est_restricted4.summary())

y=rx5
est_restricted5=sm.OLS(y, sm.add_constant(x1)).fit()
print(est_restricted5.summary())

y=rx2
x2=x['f(2,1)']-x['y1']
est_fama2=sm.OLS(y, sm.add_constant(x2)).fit()
print(est_fama2.summary())

y=rx3
x2=x['f(3,2)']-x['y1']
est_fama3=sm.OLS(y, sm.add_constant(x2)).fit()
print(est_fama3.summary())

y=rx4
x2=x['f(4,3)']-x['y1']
est_fama4=sm.OLS(y, sm.add_constant(x2)).fit()
print(est_fama4.summary())

y=rx5
x2=x['f(5,4)']-x['y1']
est_fama5=sm.OLS(y, sm.add_constant(x2)).fit()
print(est_fama5.summary())

# Term Structure

data['level']=(data['y5']+data['y4']+data['y3']+data['y2']+data['y1'])/5
data['slope']=data['y5']-data['y1']
data['curve']=2*data['y3']-data['y5']-data['y1']
y=rx_t
x_y=data[['y1','y2','y3','y4','y5']]
est_none=sm.OLS(y, sm.add_constant(x_y)).fit()
print(est_none.summary())

x_y=data[['y1','y2','y3','y4','y5','level','slope']]
est_ls=sm.OLS(y, sm.add_constant(x_y)).fit()
print(est_ls.summary())

x_y=data[['y1','y2','y3','y4','y5','level','slope','curve']]
est_lsc=sm.OLS(y, sm.add_constant(x_y)).fit()
print(est_lsc.summary())

plt.figure(figsize = (16,8))
plt.plot(rx_t,label = 'reality')
plt.plot(est_single.predict(),label = 'single factor')
plt.plot(est_lsc.predict(),label = 'term structure')
plt.legend()

# Macro Variable

plt.figure(figsize=(16,8))
data['delta M1'].loc['1997-8':].plot(legend=True,color='black',linestyle='--')
data['delta M2'].loc['1997-8':].plot(legend=True,color='black')

plt.figure(figsize=(16,8))
data['delta CPI'].loc['1997-8':].plot(legend=True)
data['delta PPI1'].loc['1997-8':].plot(legend=True)
data['delta PPI2'].loc['1997-8':].plot(legend=True)
data['delta PPI3'].loc['1997-8':].plot(legend=True)
data['delta PPI5'].loc['1997-8':].plot(legend=True)

plt.figure(figsize=(16,8))
data['delta CPI'].loc['1997-8':].plot(legend=True,color='black')
data['delta PPI3'].loc['1997-8':].plot(legend=True,color='black',linestyle='--')

plt.figure(figsize = (16,8))
data['delta C'].loc['1997-8':].plot(legend=True)

rx2=data['h(2,1)']-data['y1']
rx3=data['h(3,2)']-data['y1']
rx4=data['h(4,3)']-data['y1']
rx5=data['h(5,4)']-data['y1']
data['CPI']=data['delta CPI']
data['C']=data['delta C']
data['M1_']=data['delta M1']
data['M2_']=data['delta M2']
data['PPI']=data['delta PPI3']

y=rx2.loc['1997-8':]
x=data[['r(1,t)','PPI','M1_','M2_']].loc['1997-8':]
est = sm.OLS(y, sm.add_constant(x)).fit()
print(est.summary())

y=rx3.loc['1997-8':]
x=data[['r(1,t)','PPI','M1_','M2_']].loc['1997-8':]
est = sm.OLS(y, sm.add_constant(x)).fit()
print(est.summary())

y=rx4.loc['1997-8':]
x=data[['r(1,t)','PPI','M1_','M2_']].loc['1997-8':]
est = sm.OLS(y, sm.add_constant(x)).fit()
print(est.summary())

y=rx5.loc['1997-8':]
x=data[['r(1,t)','PPI','M1_','M2_']].loc['1997-8':]
est = sm.OLS(y, sm.add_constant(x)).fit()
print(est.summary())

y=rx2.loc['1997-8':]
x=data[['y1','f(2,1)','f(3,2)','f(4,3)','f(5,4)','PPI','M1_','M2_']].loc['1997-8':]
est = sm.OLS(y, sm.add_constant(x)).fit()
print(est.summary())

y=rx3.loc['1997-8':]
x=data[['y1','f(2,1)','f(3,2)','f(4,3)','f(5,4)','PPI','M1_','M2_']].loc['1997-8':]
est = sm.OLS(y, sm.add_constant(x)).fit()
print(est.summary())

y=rx4.loc['1997-8':]
x=data[['y1','f(2,1)','f(3,2)','f(4,3)','f(5,4)','PPI','M1_','M2_']].loc['1997-8':]
est = sm.OLS(y, sm.add_constant(x)).fit()
print(est.summary())

y=rx5.loc['1997-8':]
x=data[['y1','f(2,1)','f(3,2)','f(4,3)','f(5,4)','PPI','M1_','M2_']].loc['1997-8':]
est = sm.OLS(y, sm.add_constant(x)).fit()
print(est.summary())

y=rx_t.loc['1997-8':]
x=data[['y1','f(2,1)','f(3,2)','f(4,3)','f(5,4)','PPI','M1_','M2_']].loc['1997-8':]
est = sm.OLS(y, sm.add_constant(x)).fit()
print(est.summary())

y=rx_t.loc['1997-8':]
x = data[['y1','f(2,1)','f(3,2)','f(4,3)','f(5,4)']].loc['1997-8':]
est_single_1 = sm.OLS(y, sm.add_constant(x)).fit()
print(est_single_1.summary())

x_y=data[['y1','y2','y3','y4','y5','level','slope','curve']].loc['1997-8':]
est_lsc_1=sm.OLS(y, sm.add_constant(x_y)).fit()
print(est_lsc_1.summary())

total_df = pd.DataFrame(index = data.loc['1997-8':].index,columns = ['single','macro'])
total_df['single'] = est_single_1.predict()
total_df['macro'] = est.predict()
plt.figure(figsize = (16,8))
plt.plot(rx_t.loc['1997-8':],label = 'reality')
plt.plot(total_df['single'],label = 'single factor')
plt.plot(total_df['macro'],label = 'macro')
plt.legend()

plt.figure(figsize = (16,8))
plt.scatter(rx_t.loc['1997-8':],total_df['macro'] )
plt.axhline(y=0,c = 'black')
plt.axvline(x=0,c = 'black')

plt.figure(figsize = (16,8))
plt.scatter(rx_t.loc['1997-8':],total_df['single'] )
plt.axhline(y=0,c = 'black')
plt.axvline(x=0,c = 'black')

# Strategy

y=rx2.loc['1997-8':]
x=data[['y1','f(2,1)','f(3,2)','f(4,3)','f(5,4)','PPI','M1_','M2_']].loc['1997-8':]
est_2_macro = sm.OLS(y, sm.add_constant(x)).fit()
strategy_rx2 = pd.DataFrame(index = data.loc['1997-8':].index,columns = ['single','macro','real'])
strategy_rx2['macro'] = strategy_rx2['single'] = strategy_rx2['real'] = 0
strategy_rx2['real'].iloc[0] = rx2.loc['1997-8':][0]
for i in range(1,len(strategy_rx2)):
    strategy_rx2['real'].iloc[i] = strategy_rx2['real'].iloc[i-1] + rx2.loc['1997-8':][i]
strategy_rx2['macro'].iloc[0] = rx2.loc['1997-8':][0]
for i in range(1,len(strategy_rx2)):
    if est_2_macro.predict()[i] >0 :
        strategy_rx2['macro'].iloc[i] = strategy_rx2['macro'].iloc[i-1] + rx2.loc['1997-8':][i]
    else:
        strategy_rx2['macro'].iloc[i] = strategy_rx2['macro'].iloc[i-1]
y=rx2.loc['1997-8':]
x=data[['y1','f(2,1)','f(3,2)','f(4,3)','f(5,4)']].loc['1997-8':]
est_2_single = sm.OLS(y, sm.add_constant(x)).fit()
strategy_rx2['single'].iloc[0] = rx2.loc['1997-8':][0]
for i in range(1,len(strategy_rx2)):
    if est_2_single.predict()[i] >0 :
        strategy_rx2['single'].iloc[i] = strategy_rx2['single'].iloc[i-1] + rx2.loc['1997-8':][i]
    else:
        strategy_rx2['single'].iloc[i] = strategy_rx2['single'].iloc[i-1]
plt.figure(figsize = (16,8))
plt.plot(strategy_rx2['real'],label = 'real')
plt.plot(strategy_rx2['single'],label = 'single')
plt.plot(strategy_rx2['macro'],label = 'macro')
plt.legend()



print(empyrical.annual_return((strategy_rx2 - strategy_rx2.shift(1))['real'],period = 'monthly'))
print(empyrical.annual_return((strategy_rx2 - strategy_rx2.shift(1))['single'],period = 'monthly'))
print(empyrical.annual_return((strategy_rx2 - strategy_rx2.shift(1))['macro'],period = 'monthly'))
print(empyrical.max_drawdown((strategy_rx2 - strategy_rx2.shift(1))['real']))
print(empyrical.max_drawdown((strategy_rx2 - strategy_rx2.shift(1))['single']))
print(empyrical.max_drawdown((strategy_rx2 - strategy_rx2.shift(1))['macro']))
print(empyrical.sharpe_ratio((strategy_rx2 - strategy_rx2.shift(1))['real'],risk_free = 0, period = 'monthly'))
print(empyrical.sharpe_ratio((strategy_rx2 - strategy_rx2.shift(1))['single'],risk_free = 0, period = 'monthly'))
print(empyrical.sharpe_ratio((strategy_rx2 - strategy_rx2.shift(1))['macro'],risk_free = 0, period = 'monthly'))

strategy_rx3 = pd.DataFrame(index = data.loc['1997-8':].index,columns = ['single','macro','real'])
strategy_rx3['macro'] = strategy_rx3['single'] = strategy_rx3['real'] = 0
y=rx3.loc['1997-8':]
x=data[['y1','f(2,1)','f(3,2)','f(4,3)','f(5,4)','PPI','M1_','M2_']].loc['1997-8':]
est_3_macro = sm.OLS(y, sm.add_constant(x)).fit()
strategy_rx3['real'].iloc[0] = rx3.loc['1997-8':][0]
for i in range(1,len(strategy_rx3)):
    strategy_rx3['real'].iloc[i] = strategy_rx3['real'].iloc[i-1] + rx3.loc['1997-8':][i]
strategy_rx3['macro'].iloc[0] = rx3.loc['1997-8':][0]
for i in range(1,len(strategy_rx3)):
    if est_3_macro.predict()[i] >0 :
        strategy_rx3['macro'].iloc[i] = strategy_rx3['macro'].iloc[i-1] + rx3.loc['1997-8':][i]
    else:
        strategy_rx3['macro'].iloc[i] = strategy_rx3['macro'].iloc[i-1]
y=rx3.loc['1997-8':]
x=data[['y1','f(2,1)','f(3,2)','f(4,3)','f(5,4)']].loc['1997-8':]
est_3_single = sm.OLS(y, sm.add_constant(x)).fit()
strategy_rx3['single'].iloc[0] = rx3.loc['1997-8':][0]
for i in range(1,len(strategy_rx3)):
    if est_3_single.predict()[i] >0 :
        strategy_rx3['single'].iloc[i] = strategy_rx3['single'].iloc[i-1] + rx3.loc['1997-8':][i]
    else:
        strategy_rx3['single'].iloc[i] = strategy_rx3['single'].iloc[i-1]
plt.figure(figsize = (16,8))
plt.plot(strategy_rx3['real'],label = 'real')
plt.plot(strategy_rx3['single'],label = 'single')
plt.plot(strategy_rx3['macro'],label = 'macro')
plt.legend()

print(empyrical.annual_return((strategy_rx3 - strategy_rx3.shift(1))['real'],period = 'monthly'))
print(empyrical.annual_return((strategy_rx3 - strategy_rx3.shift(1))['single'],period = 'monthly'))
print(empyrical.annual_return((strategy_rx3 - strategy_rx3.shift(1))['macro'],period = 'monthly'))
print(empyrical.max_drawdown((strategy_rx3 - strategy_rx3.shift(1))['real']))
print(empyrical.max_drawdown((strategy_rx3 - strategy_rx3.shift(1))['single']))
print(empyrical.max_drawdown((strategy_rx3 - strategy_rx3.shift(1))['macro']))
print(empyrical.sharpe_ratio((strategy_rx3 - strategy_rx3.shift(1))['real'],risk_free = 0, period = 'monthly'))
print(empyrical.sharpe_ratio((strategy_rx3 - strategy_rx3.shift(1))['single'],risk_free = 0, period = 'monthly'))
print(empyrical.sharpe_ratio((strategy_rx3 - strategy_rx3.shift(1))['macro'],risk_free = 0, period = 'monthly'))

strategy_rx4 = pd.DataFrame(index = data.loc['1997-8':].index,columns = ['single','macro','real'])
strategy_rx4['macro'] = strategy_rx4['single'] = strategy_rx4['real'] = 0
y=rx4.loc['1997-8':]
x=data[['y1','f(2,1)','f(3,2)','f(4,3)','f(5,4)','PPI','M1_','M2_']].loc['1997-8':]
est_4_macro = sm.OLS(y, sm.add_constant(x)).fit()
strategy_rx4['real'].iloc[0] = rx4.loc['1997-8':][0]
for i in range(1,len(strategy_rx4)):
    strategy_rx4['real'].iloc[i] = strategy_rx4['real'].iloc[i-1] + rx4.loc['1997-8':][i]
strategy_rx4['macro'].iloc[0] = rx4.loc['1997-8':][0]
for i in range(1,len(strategy_rx4)):
    if est_4_macro.predict()[i] >0 :
        strategy_rx4['macro'].iloc[i] = strategy_rx4['macro'].iloc[i-1] + rx4.loc['1997-8':][i]
    else:
        strategy_rx4['macro'].iloc[i] = strategy_rx4['macro'].iloc[i-1]
y=rx4.loc['1997-8':]
x=data[['y1','f(2,1)','f(3,2)','f(4,3)','f(5,4)']].loc['1997-8':]
est_4_single = sm.OLS(y, sm.add_constant(x)).fit()
strategy_rx4['single'].iloc[0] = rx4.loc['1997-8':][0]
for i in range(1,len(strategy_rx4)):
    if est_4_single.predict()[i] >0 :
        strategy_rx4['single'].iloc[i] = strategy_rx4['single'].iloc[i-1] + rx4.loc['1997-8':][i]
    else:
        strategy_rx4['single'].iloc[i] = strategy_rx4['single'].iloc[i-1]
plt.figure(figsize = (16,8))
plt.plot(strategy_rx4['real'],label = 'real')
plt.plot(strategy_rx4['single'],label = 'single')
plt.plot(strategy_rx4['macro'],label = 'macro')
plt.legend()

print(empyrical.annual_return((strategy_rx4 - strategy_rx4.shift(1))['real'],period = 'monthly'))
print(empyrical.annual_return((strategy_rx4 - strategy_rx4.shift(1))['single'],period = 'monthly'))
print(empyrical.annual_return((strategy_rx4 - strategy_rx4.shift(1))['macro'],period = 'monthly'))
print(empyrical.max_drawdown((strategy_rx4 - strategy_rx4.shift(1))['real']))
print(empyrical.max_drawdown((strategy_rx4 - strategy_rx4.shift(1))['single']))
print(empyrical.max_drawdown((strategy_rx4 - strategy_rx4.shift(1))['macro']))
print(empyrical.sharpe_ratio((strategy_rx4 - strategy_rx4.shift(1))['real'],risk_free = 0, period = 'monthly'))
print(empyrical.sharpe_ratio((strategy_rx4 - strategy_rx4.shift(1))['single'],risk_free = 0, period = 'monthly'))
print(empyrical.sharpe_ratio((strategy_rx4 - strategy_rx4.shift(1))['macro'],risk_free = 0, period = 'monthly'))

strategy_rx5 = pd.DataFrame(index = data.loc['1997-8':].index,columns = ['single','macro','real'])
strategy_rx5['macro'] = strategy_rx5['single'] = strategy_rx5['real'] = 0
y=rx5.loc['1997-8':]
x=data[['y1','f(2,1)','f(3,2)','f(4,3)','f(5,4)','PPI','M1_','M2_']].loc['1997-8':]
est_5_macro = sm.OLS(y, sm.add_constant(x)).fit()
strategy_rx5['real'].iloc[0] = rx5.loc['1997-8':][0]
for i in range(1,len(strategy_rx3)):
    strategy_rx5['real'].iloc[i] = strategy_rx5['real'].iloc[i-1] + rx5.loc['1997-8':][i]
strategy_rx5['macro'].iloc[0] = rx5.loc['1997-8':][0]
for i in range(1,len(strategy_rx5)):
    if est_5_macro.predict()[i] >0 :
        strategy_rx5['macro'].iloc[i] = strategy_rx5['macro'].iloc[i-1] + rx5.loc['1997-8':][i]
    else:
        strategy_rx5['macro'].iloc[i] = strategy_rx5['macro'].iloc[i-1]
y=rx5.loc['1997-8':]
x=data[['y1','f(2,1)','f(3,2)','f(4,3)','f(5,4)']].loc['1997-8':]
est_5_single = sm.OLS(y, sm.add_constant(x)).fit()
strategy_rx5['single'].iloc[0] = rx5.loc['1997-8':][0]
for i in range(1,len(strategy_rx5)):
    if est_5_single.predict()[i] >0 :
        strategy_rx5['single'].iloc[i] = strategy_rx5['single'].iloc[i-1] + rx5.loc['1997-8':][i]
    else:
        strategy_rx5['single'].iloc[i] = strategy_rx5['single'].iloc[i-1]
plt.figure(figsize = (16,8))
plt.plot(strategy_rx5['real'],label = 'real')
plt.plot(strategy_rx5['single'],label = 'single')
plt.plot(strategy_rx5['macro'],label = 'macro')
plt.legend()

print(empyrical.annual_return((strategy_rx5 - strategy_rx5.shift(1))['real'],period = 'monthly'))
print(empyrical.annual_return((strategy_rx5 - strategy_rx5.shift(1))['single'],period = 'monthly'))
print(empyrical.annual_return((strategy_rx5 - strategy_rx5.shift(1))['macro'],period = 'monthly'))
print(empyrical.max_drawdown((strategy_rx5 - strategy_rx5.shift(1))['real']))
print(empyrical.max_drawdown((strategy_rx5 - strategy_rx5.shift(1))['single']))
print(empyrical.max_drawdown((strategy_rx5 - strategy_rx5.shift(1))['macro']))
print(empyrical.sharpe_ratio((strategy_rx5 - strategy_rx5.shift(1))['real'],risk_free = 0, period = 'monthly'))
print(empyrical.sharpe_ratio((strategy_rx5 - strategy_rx5.shift(1))['single'],risk_free = 0, period = 'monthly'))
print(empyrical.sharpe_ratio((strategy_rx5 - strategy_rx5.shift(1))['macro'],risk_free = 0, period = 'monthly'))

strategy_rxt = pd.DataFrame(index = data.loc['1997-8':].index,columns = ['single','macro','real'])
y=rx_t.loc['1997-8':]
x=data[['y1','f(2,1)','f(3,2)','f(4,3)','f(5,4)','PPI','M1_','M2_']].loc['1997-8':]
est_t_macro = sm.OLS(y, sm.add_constant(x)).fit()
strategy_rxt['real'].iloc[0] = rx_t.loc['1997-8':][0]
for i in range(1,len(strategy_rx3)):
    strategy_rxt['real'].iloc[i] = strategy_rxt['real'].iloc[i-1] + rx_t.loc['1997-8':][i]
strategy_rxt['macro'].iloc[0] = rx_t.loc['1997-8':][0]
for i in range(1,len(strategy_rxt)):
    if est_t_macro.predict()[i] >0 :
        strategy_rxt['macro'].iloc[i] = strategy_rxt['macro'].iloc[i-1] + rx_t.loc['1997-8':][i]
    else:
        strategy_rxt['macro'].iloc[i] = strategy_rxt['macro'].iloc[i-1]
y=rx_t.loc['1997-8':]
x=data[['y1','f(2,1)','f(3,2)','f(4,3)','f(5,4)']].loc['1997-8':]
est_t_single = sm.OLS(y, sm.add_constant(x)).fit()
strategy_rxt['single'].iloc[0] = rx_t.loc['1997-8':][0]
for i in range(1,len(strategy_rxt)):
    if est_t_single.predict()[i] >0 :
        strategy_rxt['single'].iloc[i] = strategy_rxt['single'].iloc[i-1] + rx_t.loc['1997-8':][i]
    else:
        strategy_rxt['single'].iloc[i] = strategy_rxt['single'].iloc[i-1]
plt.figure(figsize = (16,8))
plt.plot(strategy_rxt['real'],label = 'real')
plt.plot(strategy_rxt['single'],label = 'single')
plt.plot(strategy_rxt['macro'],label = 'macro')
plt.legend()

strategy_buy_sell = pd.DataFrame(index = data.loc['1997-8':].index,columns = ['single','macro','real'])
strategy_buy_sell['macro'] = strategy_buy_sell['single'] = strategy_buy_sell['real'] = 0
strategy_buy_sell['real'].iloc[0] = rx_t.loc['1997-8':][0]
for i in range(1,len(strategy_rx3)):
    strategy_buy_sell['real'].iloc[i] = strategy_buy_sell['real'].iloc[i-1] + rx_t.loc['1997-8':][i]
strategy_buy_sell['macro'].iloc[0] = rx_t.loc['1997-8':][0]
for i in range(1,len(strategy_buy_sell)):
    r2 = est_2_macro.predict()[i]
    r3 = est_3_macro.predict()[i]
    r4 = est_4_macro.predict()[i]
    r5 = est_5_macro.predict()[i]
    if max(r2,r3,r4,r5) == r2:
        if max(r3,r4,r5) == r3:
            strategy_buy_sell['macro'].iloc[i] = strategy_buy_sell['macro'].iloc[i-1] + rx2.loc['1997-8':][i]+ rx3.loc['1997-8':][i]
        elif max(r3,r4,r5) == r4:
            strategy_buy_sell['macro'].iloc[i] = strategy_buy_sell['macro'].iloc[i-1] + rx2.loc['1997-8':][i]+ rx4.loc['1997-8':][i]           
        elif max(r3,r4,r5) == r5:
            strategy_buy_sell['macro'].iloc[i] = strategy_buy_sell['macro'].iloc[i-1] + rx2.loc['1997-8':][i]+ rx5.loc['1997-8':][i]            
    elif max(r2,r3,r4,r5) == r3:
        if max(r2,r4,r5) == r2:
            strategy_buy_sell['macro'].iloc[i] = strategy_buy_sell['macro'].iloc[i-1] + rx2.loc['1997-8':][i]+ rx3.loc['1997-8':][i]            
        elif max(r2,r4,r5) == r4:
            strategy_buy_sell['macro'].iloc[i] = strategy_buy_sell['macro'].iloc[i-1] + rx3.loc['1997-8':][i]+ rx4.loc['1997-8':][i]            
        elif max(r2,r4,r5) == r5:
            strategy_buy_sell['macro'].iloc[i] = strategy_buy_sell['macro'].iloc[i-1] + rx3.loc['1997-8':][i]+ rx5.loc['1997-8':][i]            
    elif max(r2,r3,r4,r5) == r4:
        if max(r2,r3,r5) == r2:
            strategy_buy_sell['macro'].iloc[i] = strategy_buy_sell['macro'].iloc[i-1] + rx2.loc['1997-8':][i]+ rx4.loc['1997-8':][i]            
        elif max(r2,r3,r5) == r3:
            strategy_buy_sell['macro'].iloc[i] = strategy_buy_sell['macro'].iloc[i-1] + rx3.loc['1997-8':][i]+ rx4.loc['1997-8':][i]           
        elif max(r2,r3,r5) == r5:
            strategy_buy_sell['macro'].iloc[i] = strategy_buy_sell['macro'].iloc[i-1] + rx4.loc['1997-8':][i]+ rx5.loc['1997-8':][i]            
    elif max(r2,r3,r4,r5) == r5:
        if max(r2,r3,r4) == r2:
            strategy_buy_sell['macro'].iloc[i] = strategy_buy_sell['macro'].iloc[i-1] + rx2.loc['1997-8':][i]+ rx5.loc['1997-8':][i]            
        elif max(r2,r3,r4) == r3:
            strategy_buy_sell['macro'].iloc[i] = strategy_buy_sell['macro'].iloc[i-1] + rx3.loc['1997-8':][i]+ rx5.loc['1997-8':][i]            
        elif max(r2,r3,r4) == r4:
            strategy_buy_sell['macro'].iloc[i] = strategy_buy_sell['macro'].iloc[i-1] + rx4.loc['1997-8':][i]+ rx5.loc['1997-8':][i]            
strategy_buy_sell['single'].iloc[0] = rx_t.loc['1997-8':][0]
for i in range(1,len(strategy_buy_sell)):
    r2 = est_2_single.predict()[i]
    r3 = est_3_single.predict()[i]
    r4 = est_4_single.predict()[i]
    r5 = est_5_single.predict()[i]
    if max(r2,r3,r4,r5) == r2:
        if max(r3,r4,r5) == r3:
            strategy_buy_sell['single'].iloc[i] = strategy_buy_sell['single'].iloc[i-1] + rx2.loc['1997-8':][i]+ rx3.loc['1997-8':][i]
        elif max(r3,r4,r5) == r4:
            strategy_buy_sell['single'].iloc[i] = strategy_buy_sell['single'].iloc[i-1] + rx2.loc['1997-8':][i]+ rx4.loc['1997-8':][i]           
        elif max(r3,r4,r5) == r5:
            strategy_buy_sell['single'].iloc[i] = strategy_buy_sell['single'].iloc[i-1] + rx2.loc['1997-8':][i]+ rx5.loc['1997-8':][i]            
    elif max(r2,r3,r4,r5) == r3:
        if max(r2,r4,r5) == r2:
            strategy_buy_sell['single'].iloc[i] = strategy_buy_sell['single'].iloc[i-1] + rx2.loc['1997-8':][i]+ rx3.loc['1997-8':][i]            
        elif max(r2,r4,r5) == r4:
            strategy_buy_sell['single'].iloc[i] = strategy_buy_sell['single'].iloc[i-1] + rx3.loc['1997-8':][i]+ rx4.loc['1997-8':][i]            
        elif max(r2,r4,r5) == r5:
            strategy_buy_sell['single'].iloc[i] = strategy_buy_sell['single'].iloc[i-1] + rx3.loc['1997-8':][i]+ rx5.loc['1997-8':][i]            
    elif max(r2,r3,r4,r5) == r4:
        if max(r2,r3,r5) == r2:
            strategy_buy_sell['single'].iloc[i] = strategy_buy_sell['single'].iloc[i-1] + rx2.loc['1997-8':][i]+ rx4.loc['1997-8':][i]            
        elif max(r2,r3,r5) == r3:
            strategy_buy_sell['single'].iloc[i] = strategy_buy_sell['single'].iloc[i-1] + rx3.loc['1997-8':][i]+ rx4.loc['1997-8':][i]           
        elif max(r2,r3,r5) == r5:
            strategy_buy_sell['single'].iloc[i] = strategy_buy_sell['single'].iloc[i-1] + rx4.loc['1997-8':][i]+ rx5.loc['1997-8':][i]            
    elif max(r2,r3,r4,r5) == r5:
        if max(r2,r3,r4) == r2:
            strategy_buy_sell['single'].iloc[i] = strategy_buy_sell['single'].iloc[i-1] + rx2.loc['1997-8':][i]+ rx5.loc['1997-8':][i]            
        elif max(r2,r3,r4) == r3:
            strategy_buy_sell['single'].iloc[i] = strategy_buy_sell['single'].iloc[i-1] + rx3.loc['1997-8':][i]+ rx5.loc['1997-8':][i]            
        elif max(r2,r3,r4) == r4:
            strategy_buy_sell['single'].iloc[i] = strategy_buy_sell['single'].iloc[i-1] + rx4.loc['1997-8':][i]+ rx5.loc['1997-8':][i]            
plt.figure(figsize = (16,8))
plt.plot(strategy_buy_sell['real'],label = 'real')
plt.plot(strategy_buy_sell['single'],label = 'single')
plt.plot(strategy_buy_sell['macro'],label = 'macro')
plt.legend()

print(empyrical.annual_return((strategy_buy_sell - strategy_buy_sell.shift(1))['real'],period = 'monthly'))
print(empyrical.annual_return((strategy_buy_sell - strategy_buy_sell.shift(1))['single'],period = 'monthly'))
print(empyrical.annual_return((strategy_buy_sell - strategy_buy_sell.shift(1))['macro'],period = 'monthly'))
print(empyrical.max_drawdown((strategy_buy_sell - strategy_buy_sell.shift(1))['real']))
print(empyrical.max_drawdown((strategy_buy_sell - strategy_buy_sell.shift(1))['single']))
print(empyrical.max_drawdown((strategy_buy_sell - strategy_buy_sell.shift(1))['macro']))
print(empyrical.sharpe_ratio((strategy_buy_sell - strategy_buy_sell.shift(1))['real'],risk_free = 0, period = 'monthly'))
print(empyrical.sharpe_ratio((strategy_buy_sell - strategy_buy_sell.shift(1))['single'],risk_free = 0, period = 'monthly'))
print(empyrical.sharpe_ratio((strategy_buy_sell - strategy_buy_sell.shift(1))['macro'],risk_free = 0, period = 'monthly'))


###
