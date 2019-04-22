# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import glob
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

def update_customer_info(tempo):
    for index in tempo.index:
        search_msisdn = tempo['MSISDN'].get_value(index,'msisdn')
        for index2 in subscriber_billing_info_msisdn_crm_imeitac.index :
            if search_msisdn == subscriber_billing_info_msisdn_crm_imeitac.get_value(index2,'msisdn'):
                print('Yes \n')
                new_total_recharge = tempo['AMOUNT'].iloc[index] + subscriber_billing_info_msisdn_crm_imeitac.get_value(index2,'total_recharge')
                if tempo['TYPE'].iloc[index] == 'RECHARGE':
                    new_number_of_recharges = subscriber_billing_info_msisdn_crm_imeitac['number_of_recharges'].iloc[index] + 1
                    new_avg_recharge_amt =  new_total_recharge/new_number_of_recharges
                    new_avg_order_gap = 0
                    new_last_order = tempo['DATE'].iloc[index]
                    new_number_of_orders =subscriber_billing_info_msisdn_crm_imeitac.get_value(index2,'number_of_orders') + 1
                    subscriber_billing_info_msisdn_crm_imeitac.set_value(index2, 'total_recharge', new_total_recharge)
                    subscriber_billing_info_msisdn_crm_imeitac.set_value(index2,'number_of_recharges', new_number_of_recharges)
                    subscriber_billing_info_msisdn_crm_imeitac.set_value(index2,'avg_recharge_amt', new_avg_recharge_amt)
                    subscriber_billing_info_msisdn_crm_imeitac.set_value(index2,'avg_order_gap',new_avg_recharge_amt)
                    #subscriber_billing_info_msisdn_crm_imeitac.set_value(index2, 'last_order', new_avg_order_gap)
                    subscriber_billing_info_msisdn_crm_imeitac.set_value(index2,'number_of_orders', new_number_of_orders)

def merge_data(info,tempo):
    recharge_info_daily = info.append(tempo)
    

msisdn_imsi_map = pd.read_csv(r"C:\Users\ezchave\Downloads\msisdn_imsi_map_30_01_2017_output.csv",error_bad_lines=False)
msisdn_imsi_map_df = msisdn_imsi_map[['msisdn','imsi']]

subscriber_billing_info = pd.read_csv(r"C:\Users\ezchave\Downloads\subscriber_billing_info_30_01_2017_output.csv",error_bad_lines=False)

subscriber_billing_info_df = subscriber_billing_info[['imsi','update_time','billing_system_date','billing_period_start_date', 'billing_period_end_date','arpu','lbc_start_date']]
subscriber_billing_info_msisdn =  pd.merge(subscriber_billing_info_df, msisdn_imsi_map_df, how='inner', on = 'imsi')

imeitac = pd.read_csv(r"C:\Users\ezchave\Downloads\imeitac.csv",error_bad_lines=False)

crm_customer_device= pd.read_csv(r"C:\Users\ezchave\Downloads\crm_customer_device_info_23_02_2017_encoded.csv",error_bad_lines=False)

subscriber_billing_info_msisdn_crm =  pd.merge(subscriber_billing_info_msisdn, crm_customer_device, how='inner', on = 'imsi')
subscriber_billing_info_msisdn_crm_imeitac =  pd.merge(subscriber_billing_info_msisdn_crm, imeitac, how='inner', on = 'tac')

subscriber_billing_info_msisdn_crm_imeitac.sort_values('msisdn',inplace=True)
    

recharge_info_daily = pd.DataFrame()
t= pd.read_csv(r"C:\Users\ezchave\Downloads\data_assignment\outputedr1\recharge_dpa_subscriber_joinededr1.log.geibfapp1.2017_01_27.csv",error_bad_lines=False)
recharge_info_daily  = recharge_info_daily.append(t)

iterations = 0
filenames = []
for filename in glob.glob(r"C:\Users\ezchave\Downloads\data_assignment\outputedr1\*.csv",recursive = True) :
    filenames.append(filename)
    iterations = iterations + 1
    

for filename in filenames :
    m = pd.read_csv(filename)
    merge_data(recharge_info_daily,m)


recharge_info_daily.sort_values('MSISDN',inplace = True)

aggregations = {
    'AMOUNT': { 
        'TOTAL_AMOUNT' : 'sum',
        'AVG_AMOUNT' : 'mean',
        'COUNT' : 'count'
    },
    'DATE': {    
        'max_date': 'max',   
        'min_date': 'min',
        #'num_days':  (datetime.datetime.strptime('max_date', '%Y-%m-%d %H:%M:%S')) - datetime.datetime.strptime('min_date', '%Y-%m-%d %H:%M:%S')
    }  
}

final = recharge_info_daily.groupby('MSISDN').agg(aggregations)

final.columns = final.columns.droplevel(0)
final['msisdn'] = final.index
subscriber_billing_info_msisdn_crm_imeitac =  pd.merge(subscriber_billing_info_msisdn_crm, final, how='inner', on = 'msisdn')


subscriber_billing_info_msisdn_crm_imeitac.to_csv(r"C:\Users\ezchave\Downloads\output.csv", sep='\t', encoding='utf-8')

k_msisdn = subscriber_billing_info_msisdn_crm_imeitac['msisdn'].values
k_total_recharge_amount = subscriber_billing_info_msisdn_crm_imeitac['TOTAL_AMOUNT'].values
k_number_of_recharges = subscriber_billing_info_msisdn_crm_imeitac['COUNT'].values
k_avg_recharge_amt = subscriber_billing_info_msisdn_crm_imeitac['AVG_AMOUNT'].values
#k_avg_order_gap = subscriber_billing_info_msisdn_crm_imeitac['avg_order_gap'].values
#k_last_order = subscriber_billing_info_msisdn_crm_imeitac['last_order'].values



'''---for total AMount VS avg amount ---'''


'''---------------determine value of K -------------------'''
clmns = ['imsi', 'msisdn','imei', 'TOTAL_AMOUNT','AVG_AMOUNT','COUNT']
subscriber_billing_info_msisdn_crm_imeitac_2 = subscriber_billing_info_msisdn_crm_imeitac[clmns]

mms = MinMaxScaler()
mms.fit(subscriber_billing_info_msisdn_crm_imeitac_2)
data_transformed = mms.transform(subscriber_billing_info_msisdn_crm_imeitac_2)

Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(data_transformed)
    Sum_of_squared_distances.append(km.inertia_)

plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()
plt.savefig('Kvalues.png')

'''--------------- make clusters -------------------'''

kmeans = KMeans(n_clusters=2, random_state=0).fit(subscriber_billing_info_msisdn_crm_imeitac_2)
labels = kmeans.labels_
subscriber_billing_info_msisdn_crm_imeitac['clusters'] = labels
clmns.extend(['clusters'])


fig = sns.lmplot('TOTAL_AMOUNT', 'AVG_AMOUNT', 
           data=subscriber_billing_info_msisdn_crm_imeitac, 
           fit_reg=False, 
           hue="clusters",  
           scatter_kws={"marker": "D", 
                        "s": 100})
plt.title('Total Amount vs AVG amount')
plt.xlabel('TOTAL_AMOUNT')
plt.ylabel('AVG_AMOUNT')
plt.savefig('totalAmountVsAvg')


''' -------------for total AMount VS count--------------------------'''


'''--------------- make clusters -------------------'''

kmeans = KMeans(n_clusters=2, random_state=0).fit(subscriber_billing_info_msisdn_crm_imeitac_2)
labels = kmeans.labels_
subscriber_billing_info_msisdn_crm_imeitac['clusters'] = labels
clmns.extend(['clusters'])


fig = sns.lmplot('TOTAL_AMOUNT', 'COUNT', 
           data=subscriber_billing_info_msisdn_crm_imeitac, 
           fit_reg=False, 
           hue="clusters",  
           scatter_kws={"marker": "D", 
                        "s": 100})
plt.title('Total Amount vs Count')
plt.xlabel('TotalAmount')
plt.ylabel('COUNT')
plt.savefig('totalAmountVsCOUNT')



'''--------------for avg amount Vs count --------------------------'''

kmeans = KMeans(n_clusters=2, random_state=0).fit(subscriber_billing_info_msisdn_crm_imeitac_2)
labels = kmeans.labels_
subscriber_billing_info_msisdn_crm_imeitac['clusters'] = labels
clmns.extend(['clusters'])


fig = sns.lmplot('AVG_AMOUNT', 'COUNT', 
           data=subscriber_billing_info_msisdn_crm_imeitac, 
           fit_reg=False, 
           hue="clusters",  
           scatter_kws={"marker": "D", 
                        "s": 100})
plt.title('averageAmount vs Count')
plt.xlabel('AVERAGE_AMOUNT')
plt.ylabel('COUNT')
plt.savefig('AvgAmountVsCOUNT')

