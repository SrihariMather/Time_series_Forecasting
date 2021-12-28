# -*- coding: utf-8 -*-
"""
#Added following updates:
# Day-wise with date    
# Month-wise with date
# Month summary ($) -- Graphical representation with labels

"""

######################################                    FORECASTING              ######################################
import pandas as pd
import numpy as np
import itertools
from statsmodels.tsa.stattools import adfuller # ADF test -- stationarity
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from pmdarima import auto_arima
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.api import SimpleExpSmoothing
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import random
import pyodbc
import calendar
import datetime


def sql_conn():
    #Establishing the connection
    conn = pyodbc.connect('Driver={SQL Server};'
                          'Server=DESKTOP-R5PURJR;'                      
                          'username = Srihari;'
                          'password = hari;'
                          'Database=withPython;'
                          'Trusted_Connection=yes;')
    sql_dat = pd.read_sql(sql="select * FROM withPython.dbo.Payment_Prod", con = conn)
    
    return sql_dat

def pre_processing(pd_dat):
    
    # reading the file
    #pd_dat =  pd.read_excel('D://Healthcare//02Jan//Payment_06Jan.xlsx')
    #pd_dat[''] = pd.to_datetime(pd_dat['DISCHARGEDATE'])
    print('pd_dat, dtypes', pd_dat.dtypes)
    
    pd_dat['DISCHARGEDATE'] = pd.to_datetime(pd_dat['DISCHARGEDATE'])
    #pd_dat['TOTALCHARGES'] = pd_dat['TOTALCHARGES'].apply(pd.to_numeric, downcast='float', errors='coerce')
    pd_dat['TOTALINSURANCEPAYMENTS'] = pd_dat['TOTALINSURANCEPAYMENTS'].apply(pd.to_numeric, downcast='float', errors='coerce')
    pd_dat['TOTALPATIENTPAYMENTS'] = pd_dat['TOTALPATIENTPAYMENTS'].apply(pd.to_numeric, downcast='float', errors='coerce')
    #pd_dat['TOTALADJUSTMENTS'] = pd_dat['TOTALADJUSTMENTS'].apply(pd.to_numeric, downcast='float', errors='coerce')
    #pd_dat['ACCOUNTBALANCE'] = pd_dat['ACCOUNTBALANCE'].apply(pd.to_numeric, downcast='float', errors='coerce')
    
    print('TOTALINSURANCEPAYMENTS', pd_dat['TOTALINSURANCEPAYMENTS'])
    
    #Numerical conversion for Amount field
    pd_dat['TOTALINSURANCEPAYMENTS'] = -1 * pd_dat['TOTALINSURANCEPAYMENTS']
    pd_dat['TOTALPATIENTPAYMENTS'] = -1 * pd_dat['TOTALPATIENTPAYMENTS']
    
    pd_dat = pd_dat.fillna(0)
    pd_dat['TOTALPAYMENTS'] = pd_dat['TOTALINSURANCEPAYMENTS'] + pd_dat['TOTALPATIENTPAYMENTS']
    pd_dat['DISCHARGEDATE'] = pd.to_datetime(pd_dat['DISCHARGEDATE'])
    
    pd_dat.sort_values(by=['DISCHARGEDATE'], inplace=True)
    
    #pd_dat.to_csv('D://Healthcare//02Jan//chk2.csv')
    
    las_date = pd_dat['DISCHARGEDATE'].iloc[-1]
    
    return pd_dat, las_date

def finclas_frcst(pd_dat, finclas_cod):
    finclas_no = pd_dat.loc[pd_dat['ORIGINALFINANCIALCLASS'] == finclas_cod]
    
    #To get the count of finclas w.r.t. date
    fin_chk = finclas_no.groupby('DISCHARGEDATE').ORIGINALFINANCIALCLASS.value_counts()
    # To get the first index-date from series
    finclas_frcst = pd.DataFrame({'date':fin_chk.index.get_level_values(0), 'count':fin_chk.values})
    #finclas_frcst
    
    finclas_frcst.set_index('date', inplace=True)
    
    #Decomposition of the series into Trend, Seasonality and Residuals
    decomp_finclas = seasonal_decompose(finclas_frcst, model='additive', freq=1).plot()
    plt.savefig(f'D://Healthcare//02Jan//Results//decomp_{finclas_cod}.jpg', dpi=300, bbox_inches='tight')
    
    #Stationarity is checked using ADF test
    #If p-value less than 0.05, then stationarity and hence non-stationary
    stationarity_chk = adfuller(finclas_frcst)
    #print(stationarity_chk[0])
    print('p-value for the series is', stationarity_chk[1])
    
    #ACF plot -- used for Moving Average component, parameter q is used for ARIMA model
    sm.graphics.tsa.plot_acf(finclas_frcst.squeeze())
    plt.savefig(f'D://Healthcare//02Jan//Results//acf_plot_{finclas_cod}.jpg', dpi=300, bbox_inches='tight')
    
    #PACF plot -- used for Auto regression and the parameter is p
    sm.graphics.tsa.plot_pacf(finclas_frcst)
    plt.savefig(f'D://Healthcare//02Jan//Results//pacf_plot_{finclas_cod}.jpg', dpi=300, bbox_inches='tight')

    finclas_cnt = finclas_frcst['count']
    finclas_cnt = finclas_cnt.astype('float32')    
    
    model = ARIMA(finclas_cnt, order=(0,1,0))
    
    no_of_forecast = 61
    model_fit = model.fit()
    #print(model_fit.summary())

    fc, se, conf = model_fit.forecast(no_of_forecast, alpha=0.05)
    print('forecasted values are', fc)
    
    forecasted_values = pd.DataFrame(fc)
    forecasted_values.columns = ['Forecasted_value']
    forecasted_values.to_csv(f'D://Healthcare//02Jan//Results//forepay{finclas_cod}.txt', header=False, index=False, sep='\t', mode='a')

def finclas(pd_dat):
    
    finclas_grp = pd_dat.groupby(['ORIGINALFINANCIALCLASS']).size()
    #print(type(finclas_grp))
    
    finclas_grp.sort_values(axis=0, ascending=False, inplace=True)
    
    finclas_df = pd.DataFrame({'finclas':finclas_grp.index, 'count':finclas_grp.values})
    finclas_df_5 = finclas_df.head(3)
    
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    
    ax.set_title('Top 3 Finclass')
    ax.set_xlabel('Finclass code')
    ax.set_ylabel('Finclass count')    
    
    finclass1 = finclas_df_5['finclas'].apply(str)
    finclass = finclass1.tolist()
    
    count = finclas_df_5['count'].tolist()
    ax.bar(finclass, count)
    #plt.show()
    plt.savefig('D://Healthcare//02Jan//Results//finclas_top3.jpg', dpi=300, bbox_inches='tight')
    
    #Extracting the top 3 finclass code
    finclas_val = [finclas_df_5.iloc[0]['finclas'], finclas_df_5.iloc[1]['finclas'],
                   finclas_df_5.iloc[2]['finclas']]
    
    for i in finclas_val:
    
        finclas_frcst(pd_dat, i)
        
def attendes_nam(pd_dat):
    
    attendee = []
    #attendes = pd.DataFrame(columns=['nam'])
    
    #attendes['nam'] = pd.DataFrame(pd_dat['ATTENDINGPROVIDERNAME'].unique()).head(30)
    atte_nam = pd.DataFrame(pd_dat['ATTENDINGPROVIDERNAME'].unique()).head(30)
    print('attendes', atte_nam)
    print('type attendes', type(atte_nam))

    attendes = atte_nam[0].to_list()
    #attendes = atte_nam.to_list()    
    
    pd_dat['ATTENDNAME'] = pd_dat.ATTENDINGPROVIDERNAME.apply(lambda x: random.choice(attendes))
    
    le = preprocessing.LabelEncoder()
    cat = list(pd_dat['ATTENDNAME'].value_counts().index)
    le.fit(cat)
    
    pd_dat['ATTND_CODE'] = le.transform(pd_dat['ATTENDNAME'])
    
    pay_attnd = pd_dat[['DISCHARGEDATE', 'ATTENDNAME', 'ATTND_CODE']]
    
    attnd_grp = pd_dat.groupby(['ATTND_CODE']).size()
    
    attnd_grp.sort_values(axis=0, ascending=False, inplace=True)
    
    attndname_df = pd.DataFrame({'attndnam':attnd_grp.index, 'count':attnd_grp.values})
    attndname_df_3 = attndname_df.head(3)
    attndname_df_3
    
    for att in attndname_df_3['attndnam']:
        nam = pay_attnd.loc[pay_attnd['ATTND_CODE'] == att, 'ATTENDNAME'].iloc[0]
        attendee.append(nam)        
    
    print('attendees final', attendee)

    attndnam = attndname_df_3['attndnam']
    
    attndname_df_3['attendesname'] = attendee
    
    attndname_df_3.drop('attndnam', axis=1, inplace=True)
    
    print('attndname_df_3', attndname_df_3)
    
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    
    ax.set_title('Top 3 Providers')
    ax.set_xlabel('Attendee Name')
    ax.set_ylabel('Attendee count')    
    
    #attndclass1 = attndname_df_3['attndnam'].apply(str)
    attndclass = attndname_df_3['attendesname'].tolist()
    
    count = attndname_df_3['count'].tolist()
    ax.bar(attndclass, count)
    #plt.show()
    plt.savefig('D://Healthcare//02Jan//Results//attndclas_top3.jpg', dpi=300, bbox_inches='tight')
    
    #Extracting the top 5 finclass code
    #finclas_val = [finclas_df_5.iloc[0]['finclas'], finclas_df_5.iloc[1]['finclas'],
    #               finclas_df_5.iloc[2]['finclas'], finclas_df_5.iloc[3]['finclas'],
    #               finclas_df_5.iloc[4]['finclas']]
    
    for i in attndnam:
        
        #print('i', i)   
        attndnam_frcst(pay_attnd, i)
        
def attndnam_frcst(pay_attnd, attnd_cod):
    attndclas_no = pay_attnd.loc[pay_attnd['ATTND_CODE'] == attnd_cod]
    
    #To get the count of finclas w.r.t. date
    attnd_chk = attndclas_no.groupby('DISCHARGEDATE').ATTND_CODE.value_counts()
    # To get the first index-date from series
    attndclas_frcst = pd.DataFrame({'date':attnd_chk.index.get_level_values(0), 'count':attnd_chk.values})
    #finclas_frcst
    
    attndclas_frcst.set_index('date', inplace=True)
    
    #Decomposition of the series into Trend, Seasonality and Residuals
    decomp_attndclas = seasonal_decompose(attndclas_frcst, model='additive', freq=1).plot()
    plt.savefig(f'D://Healthcare//02Jan//Results//decomp_{attnd_cod}.jpg', dpi=300, bbox_inches='tight')
    
    #Stationarity is checked using ADF test
    #If p-value less than 0.05, then stationarity and hence non-stationary
    stationarity_chk = adfuller(attndclas_frcst)
    #print(stationarity_chk[0])
    print('p-value for the series is', stationarity_chk[1])
    
    #ACF plot -- used for Moving Average component, parameter q is used for ARIMA model
    sm.graphics.tsa.plot_acf(attndclas_frcst.squeeze())
    plt.savefig(f'D://Healthcare//02Jan//Results//acf_plot_{attnd_cod}.jpg', dpi=300, bbox_inches='tight')
    
    #PACF plot -- used for Auto regression and the parameter is p
    sm.graphics.tsa.plot_pacf(attndclas_frcst)
    plt.savefig(f'D://Healthcare//02Jan//Results//pacf_plot_{attnd_cod}.jpg', dpi=300, bbox_inches='tight')

    attndclas_cnt = attndclas_frcst['count']
    attndclas_cnt = attndclas_cnt.astype('float32')    
    
    model = ARIMA(attndclas_cnt, order=(0,1,0))
    
    no_of_forecast = 61
    model_fit = model.fit()
    #print(model_fit.summary())

    fc, se, conf = model_fit.forecast(no_of_forecast, alpha=0.05)
    print('forecasted values are', fc)
    
    forecasted_values = pd.DataFrame(fc)
    forecasted_values.columns = ['Forecasted_value']
    forecasted_values.to_csv(f'D://Healthcare//02Jan//Results//foreattnd{attnd_cod}.txt', header=False, index=False, sep='\t', mode='a')
    

def pymnt_frcst(pd_dat):
    frcst_period = 30
    #Logic to consolidate -- 'TOTALPAYMENTS' 
    totpay_cons = pd_dat.groupby(by='DISCHARGEDATE').agg({'TOTALPAYMENTS': 'sum'}).reset_index()
    totpay_cons.set_index('DISCHARGEDATE', inplace=True)
    print('totpay_cons', totpay_cons)
    
    #Decomposition of the series into Trend, Seasonality and Residuals
    decomp_pay = seasonal_decompose(totpay_cons, model='additive', freq=1).plot()
    plt.savefig('D://Healthcare//02Jan//Results//decomp_pymnt_frcst.jpg', dpi=300, bbox_inches='tight')
    
    #Stationarity is checked using ADF test
    stationarity_chk = adfuller(totpay_cons)
    print('p-value for the series is', stationarity_chk[1])
    
    #ACF plot -- used for Moving Average component, parameter q is used for ARIMA model
    sm.graphics.tsa.plot_acf(totpay_cons.squeeze())
    plt.savefig('D://Healthcare//02Jan//Results//acf_plot_pymnt_frcst.jpg', dpi=300, bbox_inches='tight')
    
    #PACF plot -- used for Auto regression and the parameter is p
    sm.graphics.tsa.plot_pacf(totpay_cons)    
    plt.savefig(f'D://Healthcare//02Jan//Results//pacf_plot_pymnt_frcst.jpg', dpi=300, bbox_inches='tight')
    
    #Logic to consolidate -- 'TOTALCHARGES'
    totchar_cons = pd_dat.groupby(by='DISCHARGEDATE').agg({'TOTALCHARGES': 'sum'}).reset_index()
    totchar_cons.set_index('DISCHARGEDATE', inplace=True)
    #totchar_cons
    
    #Logic to consolidate -- 'TOTALADJUSTMENTS'
    totadj_cons = pd_dat.groupby(by='DISCHARGEDATE').agg({'TOTALADJUSTMENTS': 'sum'}).reset_index()
    totadj_cons.set_index('DISCHARGEDATE', inplace=True)
    #totadj_cons
    
    #Logic to consolidate -- 'ACCOUNTBALANCE'
    totaccbal_cons = pd_dat.groupby(by='DISCHARGEDATE').agg({'ACCOUNTBALANCE': 'sum'}).reset_index()
    totaccbal_cons.set_index('DISCHARGEDATE', inplace=True)
    #totaccbal_cons
    
    pay_frcst = pd.concat([totpay_cons, totchar_cons, totadj_cons, totaccbal_cons],axis=1,sort=False).reset_index()
    pay_frcst
    
    exog = ['TOTALCHARGES', 'TOTALADJUSTMENTS', 'ACCOUNTBALANCE']
    exog_data = pay_frcst[exog]
    
    las_date = pd_dat['DISCHARGEDATE'].iloc[-1]
    las_mon = las_date.month
    las_mon = int(las_mon)
    las_mon = las_mon+1
    las_yr = las_date.year
    las_yr = int(las_yr)
    start_dat, end_dat = calendar.monthrange(las_yr, las_mon)
    start_dat
    end_dat

    sd = datetime.datetime(las_yr, las_mon, start_dat)
    strt_dat = sd.date()
    print('strt_dat', strt_dat)

    ed = datetime.datetime(las_yr, las_mon, end_dat)
    end_dat = ed.date()
    print('end_dat', end_dat)
    
    usb = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    index = pd.date_range(start = strt_dat, end = end_dat, freq=usb)
    #print(index)
    len_indx = len(index)
    print(len_indx)

    ls_indx = list(index)
    ls_indx
    
    model = auto_arima(pay_frcst['TOTALPAYMENTS'], exogenous = exog_data, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12, start_P=0, seasonal=True)             
   
    future_forecast = model.predict(n_periods=len_indx, exogenous = exog_data[-len_indx:])
    
    print('Prediceted values are', future_forecast)
    
    ls_frcst = future_forecast.tolist()
    aggr_data_df = pd.DataFrame(ls_indx, ls_frcst)
    aggr_data_df.reset_index(inplace=True)
    aggr_data_df.columns = ['Payment Forecasted', 'Date']
    columns_titles = ['Date','Payment Forecasted']
    aggr_data_df = aggr_data_df.reindex(columns=columns_titles)
    print(aggr_data_df)

    file1 = open('D://Healthcare//02Jan//forepay.txt', "a")
    aggr_data_df.to_csv('forepay.txt', index=False, mode='a')
        
    
def main():
    
    pd_dat = sql_conn()
    #print(sql_dat)
    #sql_dat.to_csv('D://Healthcare//02Jan//chk2.csv')
    
    pd_dat, las_date = pre_processing(pd_dat)
    print('las_date', las_date)
    
    finclas_df_5 = finclas(pd_dat)
    print(finclas_df_5)
    
    attendes_nam(pd_dat)
    
    pymnt_frcst(pd_dat)
    
if __name__ == "__main__":
    main()
