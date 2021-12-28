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


def pre_processing():
    
    # reading the file
    pd_dat =  pd.read_excel('D://Healthcare//02Jan/Payment_06Jan.xlsx')
    
    #Numerical conversion for Amount field
    pd_dat['TOTALINSURANCEPAYMENTS'] = -1 * pd_dat['TOTALINSURANCEPAYMENTS']
    pd_dat['TOTALPATIENTPAYMENTS'] = -1 * pd_dat['TOTALPATIENTPAYMENTS']
    
    pd_dat = pd_dat.fillna(0)
    pd_dat['TOTALPAYMENTS'] = pd_dat['TOTALINSURANCEPAYMENTS'] + pd_dat['TOTALPATIENTPAYMENTS']
    pd_dat['DISCHARGEDATE'] = pd.to_datetime(pd_dat['DISCHARGEDATE'])
    
    pd_dat.sort_values(by=['DISCHARGEDATE'], inplace=True)
    
    return pd_dat

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
    finclas_df_5 = finclas_df.head(5)
    
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    
    ax.set_title('Top 5 Finclass')
    ax.set_xlabel('Finclass code')
    ax.set_ylabel('Finclass count')    
    
    finclass1 = finclas_df_5['finclas'].apply(str)
    finclass = finclass1.tolist()
    
    count = finclas_df_5['count'].tolist()
    ax.bar(finclass, count)
    #plt.show()
    plt.savefig('D://Healthcare//02Jan//Results//finclas_top5.jpg', dpi=300, bbox_inches='tight')
    
    #Extracting the top 5 finclass code
    finclas_val = [finclas_df_5.iloc[0]['finclas'], finclas_df_5.iloc[1]['finclas'],
                   finclas_df_5.iloc[2]['finclas'], finclas_df_5.iloc[3]['finclas'],
                   finclas_df_5.iloc[4]['finclas']]
    
    for i in finclas_val:
    
        finclas_frcst(pd_dat, i)
        
def attendes_nam(pd_dat):
    
    attendee = []
    
    attendes = ['SCHULTZ, CARLA M', 'CARDA, DUSTIN J', 'EVERS, KIMBERLY M', 'LASSIG, AMY ANNE D', 'LARSEN, SANGEETHA', 'GOLDBAUM, ANDREA SEFKIN', 'BOURNE, MEREDITH S', 
    'RICHARDSON, CHAD J', 'STIRLING, JOHANNA', 'RAUSCH, DOUGLAS J', 'PEZZELLA, KELLY E', 'KARSTEN, MICHELLE L', 'RAMAN, NATARAJAN', 'DAVYDOV, BORIS', 
    'KINZIE, SPENCER D', 'BAKANOWSKI, RACHEL L', 'LEVOIR, CLAIRE C', 'MORRISON, VICKI A', 'GUNSELMAN, ERIN L', 'MATLOCK, ROBERT J', 'SILBERT, SETH C', 
    'KENT, ANNE E', 'SCHWARTZ, IAN', 'NUSBAUM, ASHLEY L', 'BART, BRUCE J', 'REMINGTON, ANNE M', 'KIMITCH, MONICA KATHERINE', 'KROOK, JON C', 'BOBBITT, KIMBERLY L',
    'GORR, HAESHIK S', 'CONTAG, STEPHANIE J', 'ODLAND, RICK M', 'FARAH, KHALIL', 'SHAUGHNESSY, MEGAN K', 'CHANG, AMY', 'ANWAR, MARIAM', 'LEATHERMAN, JAMES W',
    'MITCHELL, ERICA', 'CASTILLO, NICHOLE T', 'FISH, LISA', 'MERCIL, EMILY J', 'BACHOUR, FOUAD A', 'GOODROAD, BRIAN K', 'OEDING, MELISSA JEAN', 'LINZIE, BRADLEY M',
    'MOSCANDREW, MARIA E', 'MALLI, AHMAD H', 'XIE, SHIRLEE XIAOPING', 'STRAND, CYNTHIA M', 'TUPAKULA RAMESH, PRAVEEN', 'GONZALEZ, HERNANDO J', 'DEWITT, BRANDON D',
    'POTTS, JEROME F', 'COUNCILMAN, DAVID L', 'MARKOWSKI, RICHARD J', 'POWELL, JESSE G', 'KHOWAJA, AMEER', 'SEIEROE, MARY E', 'ZERIS, STAMATIS', 'LAFAVE, LAURA T',
    'BOMMAKANTI, SATYA V.', 'TYLER, MICHELLE M', 'VAN CLEVE, LAURA M', 'JOHNSON, BARBARA A', 'GOAD, ERIC W', 'CARLSON, MICHELLE D', 'WANG, CONNIE J', 
    'DOKKEN, BROOKE A', 'KEMPAINEN, SARAH E', 'GALICICH, WALTER ERNST', 'BUNDLIE, SCOTT R', 'KORETH, RACHEL', 'KSIONSKI, SEBASTIAN J', 'PETERSEN, KIMBERLY M',
    'PEINE, CRAIG J', 'EIDMAN, KEITH E', 'ZERA, RICHARD T', 'BERGMAN, THOMAS A', 'ORTMAN, SYNDAL ANN', 'SWEETSER, PHILIP M', 'FEIA, KENDALL J', 'HARLOW, MICHAEL C',
    'SKOVLUND, SANDRA M', 'ESTRIN, JEFFREY ISAAC', 'KARIM, REHAN M', 'AYANA, DANIEL A', 'POGEMILLER, LINDSEY L', 'KERZNER, LAWRENCE J', 'HARLOW, MICHAEL C']
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
    
    ax.set_title('Top 5 Attendees Name')
    ax.set_xlabel('Attendee Name')
    ax.set_ylabel('Attendee count')    
    
    #attndclass1 = attndname_df_3['attndnam'].apply(str)
    attndclass = attndname_df_3['attendesname'].tolist()
    
    count = attndname_df_3['count'].tolist()
    ax.bar(attndclass, count)
    #plt.show()
    plt.savefig('D://Healthcare//02Jan//Results//attndclas_top5.jpg', dpi=300, bbox_inches='tight')
    
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
    #Logic to consolidate 'TOTALPAYMENTS' 
    pay_frcst = pd_dat.groupby(by='DISCHARGEDATE').agg({'TOTALPAYMENTS': 'sum'}).reset_index()
    pay_frcst.set_index('DISCHARGEDATE', inplace=True)
    
    #Decomposition of the series into Trend, Seasonality and Residuals
    decomp_pay = seasonal_decompose(pay_frcst, model='additive', freq=1).plot()
    plt.savefig('D://Healthcare//02Jan//Results//decomp_pymnt_frcst.jpg', dpi=300, bbox_inches='tight')
    
    #Stationarity is checked using ADF test
    stationarity_chk = adfuller(pay_frcst)
    print('p-value for the series is', stationarity_chk[1])
    
    #ACF plot -- used for Moving Average component, parameter q is used for ARIMA model
    sm.graphics.tsa.plot_acf(pay_frcst.squeeze())
    plt.savefig('D://Healthcare//02Jan//Results//acf_plot_pymnt_frcst.jpg', dpi=300, bbox_inches='tight')
    
    #PACF plot -- used for Auto regression and the parameter is p
    sm.graphics.tsa.plot_pacf(pay_frcst)    
    plt.savefig(f'D://Healthcare//02Jan//Results//pacf_plot_pymnt_frcst.jpg', dpi=300, bbox_inches='tight')
    
    model = ARIMA(pay_frcst['TOTALPAYMENTS'], order=(1,0,1))
    
    no_of_forecast = 61
    model_fit = model.fit()
    #print(model_fit.summary())

    fc, se, conf = model_fit.forecast(no_of_forecast, alpha=0.05)
    print('forecasted values are', fc)
    forecasted_values = pd.DataFrame(fc)
    forecasted_values.columns = ['Forecasted_value']
    forecasted_values.to_csv('D://Healthcare//02Jan//Results//pymnt_forecast.txt', header=False, index=False, sep='\t', mode='a')
    
    
def main():
    
    pd_dat = pre_processing()
    #print(pd_dat)
    
    finclas_df_5 = finclas(pd_dat)
    #print(finclas_df_5)
    
    attendes_nam(pd_dat)
    
    pymnt_frcst(pd_dat)
    
if __name__ == "__main__":
    main()