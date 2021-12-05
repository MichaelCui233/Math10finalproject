#!/usr/bin/env python
# coding: utf-8

# The Standard and Poor's 500, or simply the S&P 500, is a stock market index tracking the performance of 500 large companies listed on stock exchanges in the United States. It is one of the most commonly followed equity indices. Ref:https://en.wikipedia.org/wiki/S%26P_500
# 
# I will clean dataset and make charts using Altair in this homework. 
# 
# Link to the data: https://www.marketwatch.com/investing/index/spx/download-data?startDate=11/22/2020&endDate=11/22/2021

# Update: I realize that one-year dataset is too small, so I manually make a 5-year dataset from 11//2016-11/22/2021 since the website only allows one-year data download at one time. 

# In[1]:


import altair as alt
import pandas as pd
import streamlit as st
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress most warnings in TensorFlow
pd.set_option('display.float_format', lambda x: '%.3f' % x)
from tensorflow import keras
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly

# Below is S&P 500 Index data of 5 year from 11/22/2016 to 11/22/2021

# In[2]:
st.title("Explore data of S&P 500")
st.write("The Standard and Poor's 500, or simply the S&P 500, is a stock market index tracking the performance of 500 large companies listed on stock exchanges in the United States. It is one of the most commonly followed equity indices. Ref:https://en.wikipedia.org/wiki/S%26P_500")
st.subheader("Dataframe")
st.write("Below is S&P 500 Index data of 5 year from 11/15/2016 to 11/22/2021 Ref:https://www.marketwatch.com/investing/index/spx/download-data?startDate=11/22/2020&endDate=11/22/2021")

df = pd.read_csv("Download Data - INDEX_US_S&P US_SPX.csv", na_values = " ")


# In[3]:


df.head()


# There are comma in number

# In[4]:


#df.dtypes


# The data type is object. First I want to get rid of comma in number

# In[5]:


df=df.applymap(lambda x: x.replace(',',''))
df.head()


# Now I need to change data type to float

# In[6]:


def can_be_numeric(c):
    try:
        pd.to_numeric(df[c])
        return True
    except:
        return False


# In[7]:


good_cols=[c for c in df.columns if can_be_numeric(c)]
df[good_cols]= df[good_cols].apply(pd.to_numeric,axis=0)


# In[8]:


#df.dtypes
st.dataframe(df)

# Dataset is ready to go. I will first show charts of all the 4 columns on the time series

# In[9]:


chartlist=[alt.Chart(df).mark_trail().encode(
    x='Date:T',
    y=c
) for c in good_cols]


# In[10]:


alt.hconcat(*chartlist)


# We can see that all four charts have an upward trend. 

# Now I want to put 'Open' and 'Close' together, 'High' and 'Low' together. 

# In[11]:


Chart_open=alt.Chart(df).mark_trail(color='yellow').encode(
    x='Date:T',
    y=alt.Y('Open',scale=alt.Scale(domain=[2000,5000]))
    
)


# In[12]:


Chart_close=alt.Chart(df).mark_trail().encode(
    x='Date:T',
    y=alt.Y('Close',scale=alt.Scale(domain=[2000,5000]))
)


# In[13]:





# In[14]:


Chart_high=alt.Chart(df).mark_trail(color='green').encode(
    x='Date:T',
    y=alt.Y('High',scale=alt.Scale(domain=[2000,5000]))
    
)
Chart_low=alt.Chart(df).mark_trail(color='red').encode(
    x='Date:T',
    y=alt.Y('Low',scale=alt.Scale(domain=[2000,5000]))
)




# In[15]:


# Conclusion:The four charts are in same shape, which means the 4 columns have very little difference, except 'High' and 'Low' are relatively different.

# I will make a formal Candlestick Chart to have a clear look of S&P 500. The thick bar represents the opening and closing prices, while the thin bar shows high and low prices; if the index closed higher on a given day, the bars are colored green rather than red. Ref: http://mbostock.github.io/protovis/ex/candlestick.html

# In[16]:
st.subheader("Candlestick Chart")
st.write("A Candlestick Chart has a clear look of S&P 500 data. The thick bar represents the opening and closing prices, while the thin bar shows high and low prices; if the index closed higher on a given day, the bars are colored green rather than red. Ref: http://mbostock.github.io/protovis/ex/candlestick.html")
st.caption("Feel free to drag or zoom in on the chart")
open_close_color = alt.condition("datum.Open <= datum.Close",
                                 alt.value("#06982d"),
                                 alt.value("#ae1325"))

base = alt.Chart(df).encode(
    alt.X('Date:T',
          axis=alt.Axis(
              format='%m/%d',
              labelAngle=-45,
              title='S&P 500'
          )
    ),
    color=open_close_color,
    tooltip=['Date:T','Open','Close','High','Low']
).properties(
    width=800,
    height=300
)

rule = base.mark_rule().encode(
    alt.Y(
        'Low:Q',
        title='Price',
        scale=alt.Scale(zero=False),
    ),
    alt.Y2('High:Q')
)

bar = base.mark_bar().encode(
    alt.Y('Open:Q'),
    alt.Y2('Close:Q')
)
st.altair_chart((rule+bar).interactive())


st.markdown("* This chart contains sophsticated information, and is hard to get a conclusion directly. ")
st.markdown("* However, one can still find that there are some very long bars in the first half of 2020. The long bars mean big variance in the same day price, which is obviously due to COVID 19.")
# The Candlestick Chart itself is not enough to analyze and find out the trend hiden under this stock. Here I will compute 50-day, 100-day, and 200-day moving average of S&P 500 using its close price. n-day moving average is arithmetic mean of n number of data points. Ref: https://www.investopedia.com/terms/m/movingaverage.asp

# In[17]:
st.subheader("Moving Average")
st.write("The Candlestick Chart itself is not enough to analyze and find out the trend hiden under this stock. Here I will compute **50-day, 100-day, and 200-day moving average** of S&P 500 using its close price. n-day moving average is arithmetic mean of n number of data points.")
st.write("In finance, a moving average is a stock indicator that is commonly used in technical analysis. The reason for calculating the moving average of a stock is to help smooth out the price data by creating a constantly updated average price. Ref: https://www.investopedia.com/terms/m/movingaverage.asp")

df['rolling_mean_50']=df['Close'].rolling(50, min_periods=1).mean()
df['rolling_mean_100']=df['Close'].rolling(100, min_periods=1).mean()
df['rolling_mean_200']=df['Close'].rolling(200, min_periods=1).mean()

color_50=st.color_picker('Pick a color for 50-day moving average','#6b6ecf')
color_100=st.color_picker('Pick a color for 100-day moving average','#17becf')
color_200=st.color_picker('Pick a color for 200-day moving average','#e7ba52')

# The purple, blue, yellow lines are 50-day, 100-day, and 200-day moving average respectively. Ref: https://altairviz.github.io/gallery/scatter_with_rolling_mean.html

# In[18]:

line_50 = alt.Chart(df).mark_line(
    color=color_50,
    size=3
).transform_window(
    rolling_mean_50='mean(Close)',
    frame=[-25, 25]
).encode(
    x='Date:T',
    y='rolling_mean_50:Q',
    tooltip=['rolling_mean_50','rolling_mean_100','rolling_mean_200']
).properties(
    width=800,
    height=300
)
line_100 = alt.Chart(df).mark_line(
    color=color_100,
    size=3
).transform_window(
    rolling_mean_100='mean(Close)',
    frame=[-50, 50]
).encode(
    x='Date:T',
    y='rolling_mean_100:Q'
)
line_200 = alt.Chart(df).mark_line(
    color=color_200,
    size=3
).transform_window(
    rolling_mean_200='mean(Close)',
    frame=[-100, 100]
).encode(
    x='Date:T',
    y='rolling_mean_200:Q',

)
st.altair_chart((line_50+line_100+line_200+rule+bar).interactive())


st.markdown("* We can see that the moving average lines are not fluctuate as intensively as the daily price does. They show an average trend of the stock.") 

# In[19]:


st.altair_chart((line_50+line_100+line_200).interactive())


st.markdown("* The 50-day moving average fluctuate relatively wildly, but the 200-day is relatively mild.")
st.markdown("* If zoom in, we can also see that those lines are not smooth. The line itself is made of numerous small periodic fluctuation.")

# Now I want to use machine learning to train the dataset to predict the trend of the stock. Let me define that if the moving average today is bigger than the one yesterday, then it has an upward trend, which shows True. 

# In[20]:
st.header("Predict S&P 500")
st.write("Now I want to use Keras to train the dataset to predict the moving average of the stock. Try the tool on the left to set the model's parameters. Ref:https://www.kaggle.com/gauravduttakiit/predict-stocks-future-prices-using-ml-dl")

with st.sidebar:
    st.write("Here you can adjust some parameters for the prediction training. Ref:https://github.com/ChristopherDavisUCI/streamlit_ed/blob/main/grad_desc.py")

    learn = st.slider("What learning rate?",min_value=0.0,max_value=0.2,step=0.001, value = 0.001,format="%.3f")

    iteration = st.slider("How many iteration you want?",min_value=1,max_value=200 ,step=1, value = 20) #

good_cols=[c for c in df.columns if can_be_numeric(c)]

X_train=df[good_cols]


#%% 50

y_train=df['rolling_mean_50']

model = keras.Sequential(
    [
        keras.layers.InputLayer(input_shape = (7,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1,activation="linear")
    ]
)

model.compile(
    loss="mean_squared_error", 
    optimizer=keras.optimizers.Adam(learning_rate=learn)
)

history=model.fit(X_train,y_train,epochs=iteration,validation_split=.2)



predicted_rolling_mean_50=model.predict(X_train)
Predicted=[]
for i in predicted_rolling_mean_50:
    Predicted.append(i[0])   #Ref:https://www.kaggle.com/gauravduttakiit/predict-stocks-future-prices-using-ml-dl


df_predicted = df[['Date']]
df2_predicted=df_predicted.copy()
df2_predicted['Prediction']=Predicted
df2_predicted['rolling_mean_50']=df['rolling_mean_50']



Chart_prediction=alt.Chart(df2_predicted).mark_trail(color='grey').encode(
    x='Date:T',
    y=alt.Y('Prediction',scale=alt.Scale(domain=[2000,5000])),
    tooltip=['Date:T','Prediction','rolling_mean_50']
    
).properties(
    width=600,
    height=300
)

with st.expander("See the prediction for 50-day moving average"):
    st.caption("Grey line represents the predicted moving average")
    st.altair_chart((Chart_prediction+line_50).interactive())


#%%100

y_train=df['rolling_mean_100']

model = keras.Sequential(
    [
        keras.layers.InputLayer(input_shape = (7,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1,activation="linear")
    ]
)

model.compile(
    loss="mean_squared_error", 
    optimizer=keras.optimizers.Adam(learning_rate=learn)
)

history=model.fit(X_train,y_train,epochs=iteration,validation_split=.2)



predicted_rolling_mean_100=model.predict(X_train)
Predicted=[]
for i in predicted_rolling_mean_100:
    Predicted.append(i[0])   #Ref:https://www.kaggle.com/gauravduttakiit/predict-stocks-future-prices-using-ml-dl


df_predicted = df[['Date']]
df2_predicted=df_predicted.copy()
df2_predicted['Prediction']=Predicted
df2_predicted['rolling_mean_100']=df['rolling_mean_100']



Chart_prediction=alt.Chart(df2_predicted).mark_trail(color='grey').encode(
    x='Date:T',
    y=alt.Y('Prediction',scale=alt.Scale(domain=[2000,5000])),
    tooltip=['Date:T','Prediction','rolling_mean_100']
    
).properties(
    width=600,
    height=300
)

with st.expander("See the prediction for 100-day moving average"):
    st.caption("Grey line represents the predicted moving average")
    st.altair_chart((Chart_prediction+line_100).interactive())



#%%200
y_train=df['rolling_mean_200']

model = keras.Sequential(
    [
        keras.layers.InputLayer(input_shape = (7,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1,activation="linear")
    ]
)

model.compile(
    loss="mean_squared_error", 
    optimizer=keras.optimizers.Adam(learning_rate=learn)
)

history=model.fit(X_train,y_train,epochs=iteration,validation_split=.2)



predicted_rolling_mean_200=model.predict(X_train)
Predicted=[]
for i in predicted_rolling_mean_200:
    Predicted.append(i[0])   #Ref:https://www.kaggle.com/gauravduttakiit/predict-stocks-future-prices-using-ml-dl


df_predicted = df[['Date']]
df2_predicted=df_predicted.copy()
df2_predicted['Prediction']=Predicted
df2_predicted['rolling_mean_200']=df['rolling_mean_200']



Chart_prediction=alt.Chart(df2_predicted).mark_trail(color='grey').encode(
    x='Date:T',
    y=alt.Y('Prediction',scale=alt.Scale(domain=[2000,5000])),
    tooltip=['Date:T','Prediction','rolling_mean_200']
    
).properties(
    width=600,
    height=300
)

with st.expander("See the prediction for 200-day moving average"):
    st.caption("Grey line represents the predicted moving average")
    st.altair_chart((Chart_prediction+line_200).interactive())


# Now I will train the model to predict trend of 50-day, 100-day, and 200-day moving average. 

# In[28]:


st.markdown("* All the results of three predictions look good, but all are a little above the true value. Especially for 200-day line, it goes way farther than the original line")
st.markdown("* The limit of this method is that it only predicts based on the current time series. It cannot really 'predict' the future.")

st.subheader("Predict using Prophet")
st.write("Prophet is a procedure for forecasting time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. It works best with time series that have strong seasonal effects and several seasons of historical data. Prophet is robust to missing data and shifts in the trend, and typically handles outliers well. Ref:https://facebook.github.io/prophet/")
st.write("I will use Prophet to predict close price in next year. Ref:https://www.kaggle.com/janiobachmann/s-p-500-time-series-forecasting-with-prophet/notebook ")

# I will do a simple prediction using only 'Close'. Before using Prophet, I need to replace column names to fit the df to Prophet.

# In[29]:


m = Prophet()

ph_df=df.loc[:,['Date','Close']]

ph_df.rename(columns={'Close': 'y', 'Date': 'ds'}, inplace=True)

ph_df.head()


# Now I will predict the upper and lower prices of the closing price. 

# In[30]:


m = Prophet()
m.fit(ph_df)


# In[31]:


future_prices = m.make_future_dataframe(periods=365)
forecast = m.predict(future_prices)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[32]:


st.plotly_chart(plot_plotly(m,forecast))

st.markdown("* The blue area is the predicted result. 'yhat_lower' and 'yhat_upper' are uncertainty intervals. The stock will probably have an upper trend next year.")

st.write("Here is the trend, yearly seasonality, and weekly seasonality. ")

# In[33]:

st.plotly_chart(plot_components_plotly(m, forecast))


st.markdown("* The stock has an overall upper trend signs yearly.")
st.markdown("* No weekly trend for stock prices.")
st.markdown("* There is a periodic flucuation on monthly prices. Upper trend during April and November. ")


st.balloons()

if st.button('Go to My GitHub repository'):
    st.write("https://github.com/MichaelCui233/Math10finalproject.git")

