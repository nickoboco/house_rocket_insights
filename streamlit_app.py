#importing packages
import pandas as pd
import streamlit as st
import numpy as np
from PIL import Image
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster

#page configuration
#st.set_page_config(layout='wide')
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.options.mode.chained_assignment = None
@st.cache(allow_output_mutation=True)

def get_data (data):
    #reading data
    data = pd.read_csv(data)
    return data

def dataframe_show (data):
    st.subheader("Base de dados")
    st.write("A tabela abaixo representa a base completa utilizada para essa análise. É possível filtrar as colunas da tabela, selecionar " \
             "Zip codes específicos, ordenar por qualquer coluna e exportar como arquivo .csv.")

    if st.checkbox('Marque para exibir a tabela', key='maintable'):  
        
        col1, col2 = st.columns(2)
        with col1:
            f_attributes = st.multiselect('Selecione as colunas', data.columns)
        with col2:
            f_zipcode = st.multiselect('Filtre o Zip code', data['zipcode'].unique())

        if (f_zipcode != []) & (f_attributes != []):
            data = data.loc[data['zipcode'].isin(f_zipcode), f_attributes]
        elif (f_zipcode != []) & (f_attributes == []):
            data = data.loc[data['zipcode'].isin(f_zipcode), :]
        elif (f_zipcode == []) & (f_attributes != []):
            data = data.loc[:, f_attributes]
        else:
            data = data.copy()

        st.dataframe(data)
        st.write("{} colunas, {} linhas".format(data.shape[1], data.shape[0]))

        data_csv = exportCsv(data)
        st.download_button(
            label="Download do relatório como CSV",
            data=data_csv,
            file_name='Recommendation_Report_Buy.csv',
            mime='text/csv')
    
    return None


def portifolio_density(data):
    # Mapa
    st.title('🌎 Mapa')
    st.subheader('Exploração do portifólio')
    st.write('O mapa abaixo apresenta todo o portifólio da base e com isso é possível analisar a distribuição de imóveis por região. ' \
             'Ao clicar no imóvel, é exibido as principais caracteristicas dele.')
    df = data

    if st.checkbox('Marque para exibir o mapa', key='exploring'):  

        # BaseMap - Folium
        density_map = folium.Map(location=[data['lat'].mean(), data['long'].mean()], default_zoom_start=15)

        marker_cluster = MarkerCluster().add_to(density_map)

        for name, row in df.iterrows():
            folium.Marker([row['lat'], row['long']], popup='Price <b>${0}</b>.<br> Features: {2} sqft, {3} bedrooms, '
                                                        '{4} bathrooms, year built: {5}'.format(row['price'],
                                                                                                row['date'],
                                                                                                row['sqft_living'],
                                                                                                row['bedrooms'],
                                                                                                row['bathrooms'],
                                                                                                row['yr_built'])).add_to(
                marker_cluster)

        folium_static(density_map)

    return None

def firstQuestion (data):

    #grouping by median per zipcode
    df1 = data[['price', 'zipcode']].groupby('zipcode').median().reset_index()

    #merging result
    data1 = pd.merge(data, df1, on='zipcode', how='inner')[['id', 'zipcode', 'price_x', 'price_y', 'condition']]

    #recommendation
    for i in range(len(data1)):
        if (data1.loc[i, 'price_x'] <= data1.loc[i, 'price_y']) & (data1.loc [i, 'condition'] >= 4):
            data1.loc[i, 'recommendation'] = 'buy'
        else:
            data1.loc[i, 'recommendation'] = 'dont buy'

    #filtering data
    data1 = data1[data1['recommendation'] == 'buy'].reset_index(drop=True)

    #formatting condition
    for i in range(len(data1)):
        if data1.loc[i, 'condition'] <= 1:
            data1.loc[i, 'condition'] = 'bad'
        elif data1.loc[i, 'condition'] <=3:
            data1.loc[i, 'condition'] = 'regular'
        elif data1.loc[i, 'condition'] <=4:
            data1.loc[i, 'condition'] = 'good'
        else:
            data1.loc[i, 'condition'] = 'great'

    #renaming columns
    data1.columns = ['ID', 'Zip Code', 'Price', 'Price Median Zipcode', 'Condition', 'Recommendation']

    return data1

def secondQuestion (data):
    
    #extracting month
    data['month'] = pd.to_datetime(data['date']).dt.strftime('%m').astype(np.int64)

    #creating column season
    data['season'] = data['month'].apply(lambda x: 'summer' if (x >= 6) & (x <= 8)
                                        else 'autumn' if (x >= 9) & (x <= 11)
                                        else 'spring' if (x >= 3) & (x <= 5)
                                        else 'winter')

    #grouping by median per zipcode and season
    df1 = data[['price', 'zipcode', 'season']].groupby(['zipcode', 'season']).median().reset_index()

    #merging result
    data2 = pd.merge(data, df1, on=['zipcode', 'season'], how='inner')[['id', 'zipcode','season', 'price_y', 'price_x', 'condition']]

    #recommendation
    for i in range(len(data2)):
        if (data2.loc[i, 'price_x'] <= data2.loc[i, 'price_y']) & (data2.loc [i, 'condition'] >= 5):
            data2.loc[i, 'recommendation'] = 'buy'
        else:
            data2.loc[i, 'recommendation'] = 'dont buy'
    
    #filtering data
    data2 = data2[data2['recommendation'] == 'buy'].reset_index(drop=True)

    #formatting condition
    for i in range(len(data2)):
        if data2.loc[i, 'condition'] <= 1:
            data2.loc[i, 'condition'] = 'bad'
        elif data2.loc[i, 'condition'] <=3:
            data2.loc[i, 'condition'] = 'regular'
        elif data2.loc[i, 'condition'] <=4:
            data2.loc[i, 'condition'] = 'good'    
        else:
            data2.loc[i, 'condition'] = 'great'

    #creating columns sell price and profit
    data2['sell_price'] = np.nan
    data2['profit'] = np.nan

    #filling sell price column
    for i in range(len(data2)):
        if data2.loc[i, 'price_x'] > data2.loc[i, 'price_y']:
            data2.loc[i, 'sell_price'] = data2.loc[i, 'price_x'] * 1.1
        else:
            data2.loc[i, 'sell_price'] = data2.loc[i, 'price_x'] * 1.3

    #filling profit column
    for i in range(len(data2)):
        data2.loc[i, 'profit'] = data2.loc[i, 'sell_price'] - data2.loc[i, 'price_x']

    #filtering data
    data2 = data2[data2['sell_price'] <= data2['price_y']].reset_index(drop=True)

    #renaming columns
    data2.columns = ['ID', 'Zip Code', 'Season', 'Avg Price', 'Buy Price', 'Condition', 'Recommendation', 'Sell Price', 'Profit']

    return data2

def thirdQuestion (data):

    data3 = data.assign(Dif_Avg_Sell = lambda x: (x['Avg Price'] - x['Sell Price']))

    invest = data3['Buy Price'].sum()
    profit = data3['Profit'].sum()

    data3count = data3[['ID', 'Zip Code', 'Season']].groupby(['Zip Code', 'Season']).count().reset_index()
    data3sum = data3[['Profit', 'Season', 'Zip Code']].groupby(['Zip Code', 'Season']).sum().reset_index()

    data3 = pd.merge(data3count, data3sum, on=['Zip Code', 'Season'], how='inner')

    data3.columns = ['Zip Code', 'Season', 'Houses', 'Total Profit']

    return data3, invest, profit

def exportCsv (data):

    #saving dataframe to csv
    return data.to_csv(index=False).encode('utf-8')

def hypotheses1 (data):
    #creating new dataframe
    dfh1 = data[['id', 'price', 'waterfront']]

    #calculating mean
    mean_price = dfh1[dfh1['waterfront'] == 0]['price'].mean()
    mean_price_wf = dfh1[dfh1['waterfront'] == 1]['price'].mean()

    #calculating difference 
    dif_wf_notwf = (mean_price_wf - mean_price)/mean_price
    dif_wf_perc = "{:.0%}".format(dif_wf_notwf)

    #printing result
    if dif_wf_notwf > 0.3:
        result = ('✅ Hipótese validada! Casas com vista para água são em média {} mais caras que as outras.'.format(dif_wf_perc))
    else:
        result = ('❌ Hipótese invalidada! A diferença média de preço é de {}.'.format(dif_wf_perc))

    code = '''      
    
    #creating new dataframe
    dfh1 = data[['id', 'price', 'waterfront']]

    #calculating mean
    mean_price = dfh1[dfh1['waterfront'] == 0]['price'].mean()
    mean_price_wf = dfh1[dfh1['waterfront'] == 1]['price'].mean()

    #calculating difference 
    dif_wf_notwf = (mean_price_wf - mean_price)/mean_price
    dif_wf_perc = "{:.0%}".format(dif_wf_notwf)

    #result
    if dif_wf_notwf > 0.3:
        result = ('✅ Hipótese validada! Casas com vista para água são em média {} mais caras que as outras.'.format(dif_wf_perc))
    else:
        result = ('❌ Hipótese invalidada! A diferença média de preço é de {}.'.format(dif_wf_perc))
        
        '''

    return result, code

def hypotheses2 (data):
    #creating new dataset
    dfh2 = data[['id', 'price', 'yr_built']]

    #calculating mean
    mean_price = dfh2['price'].mean()
    mean_price_1955back = dfh2[dfh2['yr_built'] < 1955]['price'].mean()

    #calculating difference 
    dif_1955_forw = (mean_price_1955back - mean_price)/mean_price
    dif_1955_perc = "{:.0%}".format(dif_1955_forw)

    #printing result
    if dif_1955_forw <= -0.5:
        result = ('✅ Hipótese validada! Casas contruídas antes de 1955 são em média {} mais baratas que as demais.'.format(dif_1955_perc))
    else:
        result = ('❌ Hipótese invalidada! A diferença média de preço é de {}.'.format(dif_1955_perc))
    
    code = '''
    
    #creating new dataset
    dfh2 = data[['id', 'price', 'yr_built']]

    #calculating mean
    mean_price = dfh2['price'].mean()
    mean_price_1955back = dfh2[dfh2['yr_built'] < 1955]['price'].mean()

    #calculating difference 
    dif_1955_forw = (mean_price_1955back - mean_price)/mean_price
    dif_1955_perc = "{:.0%}".format(dif_1955_forw)

    #printing result
    if dif_1955_forw <= -0.5:
        result = ('✅ Hipótese validada! Casas contruídas antes de 1955 são em média {} mais baratas que as demais.'.format(dif_1955_perc))
    else:
        result = ('❌ Hipótese invalidada! A diferença média de preço é de {}.'.format(dif_1955_perc))
    
    '''

    return result, code

def hypotheses3 (data):
    #creating new dataframe
    dfh3 = data[['id', 'sqft_lot', 'sqft_basement']]

    ##calculating mean
    mean_size_withbasement = dfh3[dfh3['sqft_basement'] != 0]['sqft_lot'].mean()
    mean_size_nobasement = dfh3[dfh3['sqft_basement'] == 0]['sqft_lot'].mean()

    #calculating difference 
    dif_size = (mean_size_withbasement-mean_size_nobasement)/mean_size_nobasement
    dif_size_perc = "{:.0%}".format(dif_size)

    #printing answer
    if dif_size > 0.4:
        result = ('✅ Hipótese validada! Casas sem porão são em média {} maiores que as demais.'.format(dif_size_perc))
    else:
        result = ('❌ Hipótese invalidada! A diferença média de tamanho é de {}.'.format(dif_size_perc))

    code = '''

    #creating new dataframe
    dfh3 = data[['id', 'sqft_lot', 'sqft_basement']]

    ##calculating mean
    mean_size_withbasement = dfh3[dfh3['sqft_basement'] != 0]['sqft_lot'].mean()
    mean_size_nobasement = dfh3[dfh3['sqft_basement'] == 0]['sqft_lot'].mean()

    #calculating difference 
    dif_size = (mean_size_withbasement-mean_size_nobasement)/mean_size_nobasement
    dif_size_perc = "{:.0%}".format(dif_size)

    #printing answer
    if dif_size > 0.4:
        result = ('✅ Hipótese validada! Casas sem porão são em média {} maiores que as demais.'.format(dif_size_perc))
    else:
        result = ('❌ Hipótese invalidada! A diferença média de tamanho é de {}.'.format(dif_size_perc))

    '''
    return result, code

def hypotheses4 (data):
    #creating filtered dataset
    dfh4 = data[['id', 'price', 'date']]
    dfh4['year'] = dfh4['date'].dt.year

    #mean price by year
    price_by_year = dfh4[['price', 'year']].groupby('year').mean().reset_index()

    #calculating difference
    price_dif = (price_by_year.iloc[1,1] - price_by_year.iloc[0,1])/price_by_year.iloc[0,1]
    price_dif_per = "{:.0%}".format(price_dif)

    #printing result
    if price_dif > 0.1:
        result = ('✅ Hipótese validada! Cresimento médio de {} YoY'.format(price_dif_per))
    else:
        result = ('❌ Hipótese invalidada! Cresimento médio de {} YoY'.format(price_dif_per))

    code = ''' 
    
    #creating filtered dataset
    dfh4 = data[['id', 'price', 'date']]
    dfh4['year'] = dfh4['date'].dt.year

    #mean price by year
    price_by_year = dfh4[['price', 'year']].groupby('year').mean().reset_index()

    #calculating difference
    price_dif = (price_by_year.iloc[1,1] - price_by_year.iloc[0,1])/price_by_year.iloc[0,1]
    price_dif_per = "{:.0%}".format(price_dif)

    #printing result
    if price_dif > 0.1:
        result = ('✅ Hipótese validada! Cresimento médio de {} YoY'.format(price_dif_per))
    else:
        result = ('❌ Hipótese invalidada! Cresimento médio de {} YoY'.format(price_dif_per))
    
    '''

    return result, code


def hypotheses5 (data):
    #creating filtered dataframe
    dfh5 = data[data['bathrooms'] == 3][['id', 'date', 'price']].reset_index(drop=True)
    dfh5['year_month'] = dfh5['date'].dt.strftime('%Y-%m')

    #mean price by year/month
    price_by_year_month = dfh5[['price', 'year_month']].groupby('year_month').mean().reset_index()

    #calculanting difference
    price_by_year_month['price_dif'] = price_by_year_month.price.diff()
    price_by_year_month['price_dif_percent'] = price_by_year_month['price'].pct_change()
    price_by_year_month['price_dif_perc'] = price_by_year_month['price'].pct_change().apply(lambda x: "{:.2%}".format(x))

    #calculating mean difference
    mean_dif = price_by_year_month['price_dif_percent'].mean()
    mean_dif_perc = "{:.2%}".format(price_by_year_month['price_dif_percent'].mean())

    #printing result
    if mean_dif > 0.15:
        result = ('✅ Hipótese validada! Cresimento médio de {} MoM'.format(mean_dif_perc))
    else:
        result = ('❌ Hipótese invalidada! Cresimento médio de {} MoM na média.'.format(mean_dif_perc))

    code = ''' 
    
    #creating filtered dataframe
    dfh5 = data[data['bathrooms'] == 3][['id', 'date', 'price']].reset_index(drop=True)
    dfh5['year_month'] = dfh5['date'].dt.strftime('%Y-%m')

    #mean price by year/month
    price_by_year_month = dfh5[['price', 'year_month']].groupby('year_month').mean().reset_index()

    #calculanting difference
    price_by_year_month['price_dif'] = price_by_year_month.price.diff()
    price_by_year_month['price_dif_percent'] = price_by_year_month['price'].pct_change()
    price_by_year_month['price_dif_perc'] = price_by_year_month['price'].pct_change().apply(lambda x: "{:.2%}".format(x))

    #calculating mean difference
    mean_dif = price_by_year_month['price_dif_percent'].mean()
    mean_dif_perc = "{:.2%}".format(price_by_year_month['price_dif_percent'].mean())

    #printing result
    if mean_dif > 0.15:
        result = ('✅ Hipótese validada! Cresimento médio de {} MoM'.format(mean_dif_perc))
    else:
        result = ('❌ Hipótese invalidada! Cresimento médio de {} MoM na média.'.format(mean_dif_perc))
     
    '''
    return result, code

def hypotheses6 (data):

    #creating filtered dataset
    dfh6 = data[['id', 'price', 'yr_renovated']]

    #mean price
    mean_price_renov = dfh6[dfh6['yr_renovated'] == 0]['price'].mean()
    mean_price_notrenov = dfh6[dfh6['yr_renovated'] != 0]['price'].mean()

    #calculating difference
    mean_price_dif = (mean_price_notrenov - mean_price_renov)/mean_price_notrenov
    mean_price_dif_perc = "{:.2%}".format(mean_price_dif)

    #printing result
    if mean_price_dif >= 0.15:
        result = ('✅ Hipótese validada! Casas reformadas são em média {} mais caras que outras.'.format(mean_price_dif_perc))
    else:
        result = ('❌ Hipótese invalidada! Deferença é de {}'.format(mean_price_dif_perc))

    code = '''

    #creating filtered dataset
    dfh6 = data[['id', 'price', 'yr_renovated']]

    #mean price
    mean_price_renov = dfh6[dfh6['yr_renovated'] == 0]['price'].mean()
    mean_price_notrenov = dfh6[dfh6['yr_renovated'] != 0]['price'].mean()

    #calculating difference
    mean_price_dif = (mean_price_notrenov - mean_price_renov)/mean_price_notrenov
    mean_price_dif_perc = "{:.2%}".format(mean_price_dif)

    #printing result
    if mean_price_dif >= 0.15:
        result = ('✅ Hipótese validada! Casas reformadas são em média {} mais caras que outras.'.format(mean_price_dif_perc))
    else:
        result = ('❌ Hipótese invalidada! Deferença é de {}'.format(mean_price_dif_perc))
    
    '''

    return result, code

def hypotheses7 (data):

    datadfh7 = data.copy()

    #extracting month
    datadfh7['month'] = pd.to_datetime(datadfh7['date']).dt.strftime('%m').astype(np.int64)
    #creating column season
    datadfh7['season'] = datadfh7['month'].apply(lambda x: 'summer' if (x >= 6) & (x <= 8)
                                        else 'autumn' if (x >= 9) & (x <= 11)
                                        else 'spring' if (x >= 3) & (x <= 5)
                                        else 'winter')

    #creating filtered dataframe
    dfh7 = datadfh7[datadfh7['waterfront'] == 1][['id', 'price', 'waterfront', 'season']]

    #mean price
    mean_price_summer = dfh7[dfh7['season'] == 'summer']['price'].mean()
    mean_price_not_summer = dfh7[dfh7['season'] != 'summer']['price'].mean()

    #calculating difference
    price_dif = (mean_price_summer-mean_price_not_summer)/mean_price_not_summer
    price_dif_perc = '{:.2%}'.format(price_dif)

    #printing result
    if price_dif >= 0.1:
        result = ('✅ Hipótese validada! Diferença de {}.'.format(price_dif_perc))
    else:
        result = ('❌ Hipótese invalidada! Diferença é de  {}.'.format(price_dif_perc))

    code = '''
    
    datadfh7 = data.copy()

    #extracting month
    datadfh7['month'] = pd.to_datetime(datadfh7['date']).dt.strftime('%m').astype(np.int64)

    #creating column season
    datadfh7['season'] = datadfh7['month'].apply(lambda x: 'summer' if (x >= 6) & (x <= 8)
                                        else 'autumn' if (x >= 9) & (x <= 11)
                                        else 'spring' if (x >= 3) & (x <= 5)
                                        else 'winter')

    #creating filtered dataframe
    dfh7 = data[data['waterfront'] == 1][['id', 'price', 'waterfront', 'season']]

    #mean price
    mean_price_summer = dfh7[dfh7['season'] == 'summer']['price'].mean()
    mean_price_not_summer = dfh7[dfh7['season'] != 'summer']['price'].mean()

    #calculating difference
    price_dif = (mean_price_summer-mean_price_not_summer)/mean_price_not_summer
    price_dif_perc = '{:.2%}'.format(price_dif)

    #printing result
    if price_dif >= 0.1:
        result = ('✅ Hipótese validada! Diferença de {}.'.format(price_dif_perc))
    else:
        result = ('❌ Hipótese invalidada! Diferença é de  {}.'.format(price_dif_perc))
        
    '''
    return result, code

def hypotheses8 (data):

    #creating filtered dataset
    dfh8 = data[['id', 'price', 'floors']]

    #mean price
    mean_price_notonefloor = dfh8[dfh8['floors'] > 1]['price'].mean()
    mean_price_onefloor = dfh8[dfh8['floors'] <= 1]['price'].mean()

    #calculating difference
    dif_price_floor = (mean_price_notonefloor-mean_price_onefloor)/mean_price_notonefloor
    dif_price_floor_per = '{:.2%}'.format(dif_price_floor)

    #printing result
    if dif_price_floor >0.3:
        result = ('✅ Hipótese validada! Diferença de {}.'.format(dif_price_floor_per))
    else:
        result = ('❌ Hipótese invalidada! Diferença é de {}.'.format(dif_price_floor_per))

    code = '''
    
    #creating filtered dataset
    dfh8 = data[['id', 'price', 'floors']]

    #mean price
    mean_price_notonefloor = dfh8[dfh8['floors'] > 1]['price'].mean()
    mean_price_onefloor = dfh8[dfh8['floors'] <= 1]['price'].mean()

    #calculating difference
    dif_price_floor = (mean_price_notonefloor-mean_price_onefloor)/mean_price_notonefloor
    dif_price_floor_per = '{:.2%}'.format(dif_price_floor)

    #printing result
    if dif_price_floor >0.3:
        result = ('✅ Hipótese validada! Diferença de {}.'.format(dif_price_floor_per))
    else:
        result = int('❌ Hipótese invalidada! Diferença é de {}.'.format(dif_price_floor_per))
    
    '''
    return result, code

def hypotheses9 (data):

    #converting sqft to sqmt
    def sqft_to_sqmt (x):
        return x * 0.09290304

    #creating filtered dataframe
    dfh9 = data[['id', 'sqft_lot', 'yr_built']]
    dfh9new = dfh9

    #mean size
    mean_sqmt_before2k = sqft_to_sqmt(dfh9[dfh9['yr_built'] < 2000]['sqft_lot'].mean())
    mean_sqmt_after2k = sqft_to_sqmt(dfh9[dfh9['yr_built'] >= 2000]['sqft_lot'].mean())

    dfh9after2k = dfh9[dfh9['yr_built'] >= 2000].reset_index(drop=True)
    dfh9after2k['sqmt_before2k'] = mean_sqmt_before2k
    dfh9after2k['sqmt_after2k'] = dfh9after2k['sqft_lot'].apply(lambda x: (sqft_to_sqmt(x)))

    dfh9after2k['dif_sqmt'] = np.nan
    for i in range(len(dfh9after2k)):
        dfh9after2k.loc[i, 'dif_sqmt'] = (dfh9after2k.loc[i, 'sqmt_before2k']- dfh9after2k.loc[i, 'sqmt_after2k'])/dfh9after2k.loc[i, 'sqmt_before2k']
        
    houses_smaller_mean = dfh9after2k[dfh9after2k['dif_sqmt'] >= -0.3]['id'].count()
    houses_total = dfh9after2k['id'].count()

    housessmaller_vs_totalhouses = houses_smaller_mean/houses_total
    housessmaller_vs_totalhouses_perc = '{:.2%}'.format(housessmaller_vs_totalhouses)

    #printing result
    if housessmaller_vs_totalhouses >= 0.7:
        result = ('✅ Hipótese validada! {} das casas construídas depois dos anos 2000 são pelo menos 30% ' \
            'menores que as demais.'.format(housessmaller_vs_totalhouses_perc))
    else:
        result = ('❌ Hipótese invalidada! Somente {} das casas construídas depois dos anos 2000 são pelo menos 30% ' \
            'menores que as demais.'.format(housessmaller_vs_totalhouses_perc))

    code = '''
    
    #converting sqft to sqmt
    def sqft_to_sqmt (x):
    return x * 0.09290304

    #creating filtered dataframe
    dfh9 = data[['id', 'sqft_lot', 'yr_built']]
    dfh9new = dfh9

    #mean size
    mean_sqmt_before2k = sqft_to_sqmt(dfh9[dfh9['yr_built'] < 2000]['sqft_lot'].mean())
    mean_sqmt_after2k = sqft_to_sqmt(dfh9[dfh9['yr_built'] >= 2000]['sqft_lot'].mean())

    dfh9after2k = dfh9[dfh9['yr_built'] >= 2000].reset_index(drop=True)
    dfh9after2k['sqmt_before2k'] = mean_sqmt_before2k
    dfh9after2k['sqmt_after2k'] = dfh9after2k['sqft_lot'].apply(lambda x: (sqft_to_sqmt(x)))

    dfh9after2k['dif_sqmt'] = np.nan
    for i in range(len(dfh9after2k)):
        dfh9after2k.loc[i, 'dif_sqmt'] = (dfh9after2k.loc[i, 'sqmt_before2k']- dfh9after2k.loc[i, 'sqmt_after2k'])/dfh9after2k.loc[i, 'sqmt_before2k']
        
    houses_smaller_mean = dfh9after2k[dfh9after2k['dif_sqmt'] >= -0.3]['id'].count()
    houses_total = dfh9after2k['id'].count()

    housessmaller_vs_totalhouses = houses_smaller_mean/houses_total
    housessmaller_vs_totalhouses_perc = '{:.2%}'.format(housessmaller_vs_totalhouses)

    #printing result
    if housessmaller_vs_totalhouses >= 0.7:
        result = ('✅ Hipótese validada! {} das casas construídas depois dos anos 2000 são pelo menos 30% ' \
            'menores que as demais.'.format(housessmaller_vs_totalhouses_perc))
    else:
        result = ('❌ Hipótese invalidada! Somente {} das casas construídas depois dos anos 2000 são pelo menos 30% ' \
            'menores que as demais.'.format(housessmaller_vs_totalhouses_perc))

    '''
    return result, code

def hypotheses10 (data):

    #creating new dataframe
    dfh10 = data[['id', 'price', 'zipcode', 'bedrooms', 'bathrooms']]
    dfh10pricezipcode = dfh10[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    dfh10 = pd.merge(dfh10, dfh10pricezipcode, on='zipcode', how='inner')

    dfh10['dif_price'] = np.nan

    for i in range(len(dfh10)):
        dfh10.loc[i, 'dif_price'] = (dfh10.loc[i, 'price_x']-
                                    dfh10.loc[i, 'price_y'])/dfh10.loc[i, 'price_y']

    dfh10new = dfh10[(dfh10['bedrooms'] >= 4)&(dfh10['bathrooms'] < 3)]
    housesbelowmean = dfh10new[dfh10new['dif_price'] < 0]['id'].count()
    housestotal = dfh10new['id'].count()

    housesbellow_vs_totalhouses = housesbelowmean/housestotal
    housesbellow_vs_totalhouses_per = '{:.2%}'.format(housesbellow_vs_totalhouses)

    #printing result
    if housesbellow_vs_totalhouses > 0.8:
        result = ('✅ Hipótese validada! {} dessas casas estão abaixo do preço médio da região.'.format(housesbellow_vs_totalhouses_per))
    else:
        result = ('❌ Hipótese invalidada! Somente {} dessas casas estão abaixo do preço médio da região.'.format(housesbellow_vs_totalhouses_per))

    code = '''
    
    #creating new dataframe
    dfh10 = data[['id', 'price', 'zipcode', 'bedrooms', 'bathrooms']]
    dfh10pricezipcode = dfh10[['price', 'zipcode']].groupby('zipcode').mean().reset_index()
    dfh10 = pd.merge(dfh10, dfh10pricezipcode, on='zipcode', how='inner')

    dfh10['dif_price'] = np.nan

    for i in range(len(dfh10)):
        dfh10.loc[i, 'dif_price'] = (dfh10.loc[i, 'price_x']-
                                    dfh10.loc[i, 'price_y'])/dfh10.loc[i, 'price_y']

    dfh10new = dfh10[(dfh10['bedrooms'] >= 4)&(dfh10['bathrooms'] < 3)]
    housesbelowmean = dfh10new[dfh10new['dif_price'] < 0]['id'].count()
    housestotal = dfh10new['id'].count()

    housesbellow_vs_totalhouses = housesbelowmean/housestotal
    housesbellow_vs_totalhouses_per = '{:.2%}'.format(housesbellow_vs_totalhouses)

    #printing result
    if housesbellow_vs_totalhouses > 0.8:
        result = ('✅ Hipótese validada! {} dessas casas estão abaixo do preço médio da região.'.format(housesbellow_vs_totalhouses_per))
    else:
        result = ('❌ Hipótese invalidada! Somente {} dessas casas estão abaixo do preço médio da região.'.format(housesbellow_vs_totalhouses_per))

    '''
    return result, code

if __name__ == "__main__":

    #data load
    path = './datasets/kc_house_data.csv'
    data = get_data(path)

    #data transformation
    #formatting date
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')

    #page title
    st.title("🏠 Projeto House Rocket")
    image = Image.open('seattleking.png')
    st.image(image, caption='King Count, Seattle, Washington, U.S.')

    #navigation options
    options = ['Sobre o projeto', 'Explorando os dados', 'Respondendo as perguntas do CEO', 'Validando insights', 'Contatos']
    navigation = st.selectbox('Navegue pelo projeto 👇', options)

    if navigation == options[0]:
        #Project objective
        st.header("📋 Sobre o projeto ")
        st.write("A empresa House Rocket decidiu investir na região de Seattle e o CEO da companhia fez alguns quesitonamentos ao "\
                "time de dados com relação a isso. Com isso em mente, o time " \
                "de dados têm agora a responsabilidade de realizar uma análise do mercado imobiliário dessa região. " \
                "Portanto, este projeto tem por objetivo identificar insights que apoiem nas tomadas de decisões, respondendo as " \
                "perguntas do CEO e maximizando o lucro da empresa.")
        st.write("Toda análise foi realizada utilizando dados públicos disponíveis no endereço abaixo: "\
                "https://www.kaggle.com/datasets/harlfoxem/housesalesprediction")
        st.write("Para esse projeto foram utilizadas as seguintes ferramentas: Python, Jupyter Notebook, Pandas, Numpy, VSCode, Streamlit")

        st.caption("⚠️ Disclaimer: O projeto a seguir é completamente fictício e utiliza dados públicos.")

    elif navigation == options[1]:
        #exploring data
        st.header("📊 Análise exploratória")
        dataframe_show(data)
        portifolio_density(data)

    elif navigation == options[2]:
        #Answers to CEO's questions
        st.header("📝 Perguntas do CEO")
        st.subheader("Quais os melhores imóveis disponíveis para compra?")
        st.write("A recomendação é procurar por imóveis em boas condições e com o preço abaixo da mediana da região.")
        st.write("Com essas premissas a quantidade de imóveis recomendados para compra é de 3863. " \
                 "Isso totalizaria um investimento de $1,521,312,984.00, que obviamente é muito alto, " \
                 "portanto, novas premissas serão adicionadas ao responder as próximas perguntas.")
        st.write("Abaixo é possível exibir a tabela somente com os imóveis recomendados para compra e fazer o download como arquivo .csv.")

        data1 = firstQuestion(data)

        if st.checkbox('Marque para exibir a tabela', key='firstquestion'):
            
            st.dataframe(data1)
            st.write("{} colunas, {} linhas".format(data1.shape[1], data1.shape[0]))

            data1_csv = exportCsv(data1)
            st.download_button(
                label="Download do relatório como CSV",
                data=data1_csv,
                file_name='Recommendation_Report_Buy.csv',
                mime='text/csv')

        st.subheader("Caso a House Rocket invista numa certa quantidade de imóveis dessa lista, qual a sugestão de compra e venda?")
        st.write("Visando maximizar o lucro e reduzir o investimento, a recomendação é procurar por imóveis na melhor condição possível e abaixo do" \
                 " preço médio da região. É importante também considerar a estação do ano em que o imóvel está sendo anunciado, pois isso" \
                 " provavelmente afeta o valor de venda, além disso, considerando um target de 30% de ROI bruto o recomendado seria selecionar" \
                 " somente imóveis que fiquem com o preço de venda abaixo do preço médio da região por estação do ano.")
        st.write("Com essas premissas a quantidade de imóveis recomendados para compra é reduzido a 241. " \
                 "Isso totalizaria um investimento de $77,541,432.00")
        st.write("Abaixo é possível exibir a tabela somente com os imóveis recomendados para compra e fazer o download como arquivo .csv.")

        data2 = secondQuestion(data)

        if st.checkbox('Marque para exibir a tabela', key='secondquestion'):
            
            st.dataframe(data2)
            st.write("{} colunas, {} linhas".format(data2.shape[1], data2.shape[0]))

            data2_csv = exportCsv(data2)
            st.download_button(
                label="Download do relatório como CSV",
                data=data2_csv,
                file_name='Recommendation_Report_Buy.csv',
                mime='text/csv')

        st.subheader("O quanto a House Rocket poderia lucrar com esse movimento?")
        data3, invest, profit = thirdQuestion(data2)
        profit = "${:0,.2f}".format(profit)

        st.write("Considerando todos os imóveis recomendados para compra o total de retorno seria de {}.".format(profit))
        st.write("Abaixo é possível exibir a tabela de lucro agrupada por região e estação do ano")

        if st.checkbox('Marque para exibir a tabela', key='thirdquestion'):

            st.dataframe(data3)

    elif navigation == options[3]:
        #Insights and hypotheses
        st.header("💡 Insights")
        
        st.subheader('H1 - Imóveis que possuem vista para água, são 30% mais caros, na média')
        result, code = hypotheses1(data)
        st.write(result)

        if st.checkbox('Marque para ver o código', key='h1'):
            st.code(code)

        st.subheader('H2 - Imóveis com data de construção menor que 1955 são 50% mais baratos na média')
        result, code = hypotheses2(data)
        st.write(result)

        if st.checkbox('Marque para ver o código', key='h2'):
            st.code(code)

        st.subheader('H3 - Imóveis sem porão possuem área total 40% maiores do que imóveis com porão')
        result, code = hypotheses3(data)
        st.write(result)

        if st.checkbox('Marque para ver o código', key='h3'):
            st.code(code)

        st.subheader('H4 - O crescimento do preço dos imóveis YoY é 10%')
        result, code = hypotheses4(data)
        st.write(result)

        if st.checkbox('Marque para ver o código', key='h4'):
            st.code(code)

        st.subheader('H5 - Imóveis com 3 banheiros tem um crescimento de MoM de 15%')
        result, code = hypotheses5(data)
        st.write(result)

        if st.checkbox('Marque para ver o código', key='h5'):
            st.code(code)

        st.subheader('H6 - Imóveis renovados são em média 15% mais caros que os não renovados')
        result, code = hypotheses6(data)
        st.write(result)

        if st.checkbox('Marque para ver o código', key='h6'):
            st.code(code)

        st.subheader('H7 - Imóveis com vista para água ficam em torno de 10% mais caros durante verão')
        result, code = hypotheses7(data)
        st.write(result)

        if st.checkbox('Marque para ver o código', key='h7'):
            st.code(code)

        st.subheader('H8 - Imóveis com mais de 1 andar são em média 30% mais caros que imóveis com 1 andar')
        result, code = hypotheses8(data)
        st.write(result)

        if st.checkbox('Marque para ver o código', key='h8'):
            st.code(code)

        st.subheader('H9 - Mais de 70% dos móveis construídos a partir do ano 2000 são em média 30% menores que os construídos antes disso')
        result, code = hypotheses9(data)
        st.write(result)

        if st.checkbox('Marque para ver o código', key='h9'):
            st.code(code)

        st.subheader('H10 - Mais de 80% dos imóveis com mais de 4 quartos que possuem menos de 3 banheiros estão abaixo da média de preço da região')
        result, code = hypotheses10(data)
        st.write(result)

        if st.checkbox('Marque para ver o código', key='h10'):
            st.code(code)

    elif navigation == options[4]:
        #contact information
        st.header("📧 Contatos")
        st.write("Me chamo Nickolas e por meio de projetos de portifólio como este estou desenvolvendo novas habilidades relacionadas ao mundo de dados." \
                 " Atuo como Gestor de Operações há mais de 7 anos e utilizo dados no dia a dia para tomada de decisão por meio de análises de KPIs e métricas " \
                 "de negócio, além disso, sou formado em Análise e " \
                 "Desenvolvimento de Sistemas e pós graduado em Ciência de Dados e meu objetivo é trabalhar como Ciêntista de Dados profissionalmente, podendo " \
                 "usar minhas habilidades em desenvolver produtos de dados como fonte de apoio aos tomadores de decisão.")
        st.write("Fique à vontade para me conhecer melhor em um dos canais abaixo ou acessar diretamente meu portifólio de projetos:")
        st.write("📬 nickolas.selhorst@gmail.com.br")
        st.write("🪪 https://www.linkedin.com/in/nickolas-selhorst/")
        st.write("📂 https://nickoboco.github.io/portifolio_projetos/")