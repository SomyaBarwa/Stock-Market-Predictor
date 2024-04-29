# django imports -------------------------------------------------------

from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response


from rest_framework import viewsets
from .models import Stock, StockPrice
from .serializers import StockSerializer, StockPriceSerializer


# ML model imports -------------------------------------------------------

import pandas_datareader as pdr
import yfinance as yf
yf.pdr_override()
import matplotlib.pyplot as plt

import pandas as pd

import numpy as np
from numpy import array

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from datetime import datetime, timedelta

# ------------------------------------------------------- Senti analysis model imports -------------------------------------------------------

# from transformers import BertTokenizer, TFBertForSequenceClassification
# from transformers import pipeline


# ------------------------------------------------------- Views -------------------------------------------------------

# Create your views here.

class StockViewSet(viewsets.ModelViewSet):
    queryset = Stock.objects.all()
    serializer_class = StockSerializer

class StockPriceViewSet(viewsets.ModelViewSet):
    queryset = StockPrice.objects.all()
    serializer_class = StockPriceSerializer



''' Usage: "http://127.0.0.1:8000/api/stockprice/AAPL/?time_step=100&future_date=30" '''
# Create a class based view that takes a stock_id and predicts the stock price
class StockPriceView(APIView):  
    def get(self, request, stock_id):   
        # Get time_step=100, future_date=30 from get request
        time_step = int(request.GET.get('time_step', 100))
        future_date = int(request.GET.get('future_date', 30))


        # Run the LSTM model only if the last_stock_data_date is older than 1 day
        stock = Stock.objects.filter(stock_id=stock_id).first()
        if stock:
            last_stock_data_date = stock.last_stock_data_date
            if last_stock_data_date:
                if (datetime.now().date() - last_stock_data_date).days < 1:
                    stock_prices = StockPrice.objects.filter(stock_id=stock)
                    serializer = StockPriceSerializer(stock_prices, many=True)
                    return Response(serializer.data)


        # Data Collection
        # Get the stock data from Yahoo Finance
        try:
            tick = yf.Ticker(stock_id.upper())

            five_years_ago = datetime.now() - timedelta(days=5*365)
            five_years_ago_str = five_years_ago.strftime('%Y-%m-%d')

            today = datetime.now()
            # today = datetime(2023, 7, 1)
            yesterday = today - timedelta(1)
            yesterday_str = yesterday.strftime('%Y-%m-%d')

            try: 
                df = pdr.data.get_data_yahoo(stock_id, start=five_years_ago_str, end=yesterday_str)
            except:
                df = tick.history(period="max")


            Stock.objects.filter(stock_id=stock_id).delete()
            stock = Stock.objects.create(stock_id=stock_id)    
            stock.last_stock_data_date = today.date()
            stock.save()

        except:
            return Response("Stock not found", status=404)
            

        # Date Preprocessing
        df_d= df
        df=df.reset_index()['Close']

        df_d['Date'] = df_d.index
        # print(df_d['Close'])

        # print("\nDone\n")

        # Date Normalization
        scaler = MinMaxScaler(feature_range=(0,1))
        df = scaler.fit_transform(np.array(df).reshape(-1,1))


        # Splitting the data into train and test
        train_len = int(len(df)*0.7)
        test_len = len(df) - train_len
        train_data, test_data = df[0:train_len,:], df[train_len:len(df),:1]


        # Convert an array of values into a dataset matrix
        def create_dataset(dataset, time_step=1):
            dataX, dataY = [], []
            for i in range(len(dataset)-time_step-1): 
                dataX.append(dataset[i:(i+time_step), 0])
                dataY.append(dataset[i + time_step, 0])
            return np.array(dataX), np.array(dataY)
        
        # Reshape into X=t,t+1,t+2,t+3 and Y=t+4
        time_step = time_step
        X_train, y_train = create_dataset(train_data, time_step)
        X_test, y_test = create_dataset(test_data, time_step)


        # Data Reshaping
        # Reshape input to be [samples, time steps, features] which is required for LSTM
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        

        # LSTM Model Creation
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
        model.add(LSTM(50, return_sequences=True))
        model.add(LSTM(50)) 
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')


        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64, verbose=1)


        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)        
        
        # Plotting
        look_back = time_step
        trainPredictPlot = np.empty_like(df)
        trainPredictPlot[:,:] = np.nan
        trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict

        testPredictPlot = np.empty_like(df)
        testPredictPlot[:,:] = np.nan
        testPredictPlot[len(train_predict)+(look_back*2)+1:len(df)-1, :] = test_predict


        # df_d= df_d[['Date', 'Close']]

        # print(df_d)
        # print(trainPredictPlot)
        # print(testPredictPlot)

        x_input = test_data[len(test_data)-time_step:].reshape(1,-1)

        temp_input = list(x_input)
        temp_input = temp_input[0].tolist()


        # Prediction
        lst_output = []
        n_steps = time_step
        i = 0

        while(i<future_date):
            if(len(temp_input)>n_steps):
                x_input = np.array(temp_input[1:])
                x_input = x_input.reshape(1, -1)
                x_input = x_input.reshape((1, n_steps, 1))
                yhat = model.predict(x_input, verbose=0)
                # Randomly add or decrease the value yhat from range of values between +yhat/30 to -yhat/30 tp add taste of stochasticity
                yhat = yhat + np.random.uniform(-yhat/30, yhat/30)

                temp_input.extend(yhat[0].tolist())
                temp_input = temp_input[1:]
            else:
                x_input = x_input.reshape((1, n_steps, 1))
                yhat = model.predict(x_input, verbose=0)
                temp_input.extend(yhat[0].tolist())

            lst_output.extend(yhat.tolist())
            i = i+1 


        past_dates = df_d[['Close', 'Date']].tail(time_step)
        print("Values used for pred: \n", past_dates)

        # Generate dates for the next 30 days, skipping Saturdays and Sundays
        future_dates = []
        day_count = 0
        while len(future_dates) < future_date:
            temp_date = today + timedelta(days=day_count)
            if temp_date.weekday() < 5:  # 0-4 denotes Monday to Friday
                future_dates.append(temp_date)
            day_count += 1

        future_dates = [date.strftime('%Y-%m-%d') for date in future_dates]

        # Convert lst_output to a DataFrame
        lst_output_df = pd.DataFrame(scaler.inverse_transform(lst_output), columns=['Predicted Close'])

        # Add future_dates as a column to lst_output_df
        lst_output_df['Date'] = future_dates
        print("Predicted values: \n", lst_output_df)   


        for index, row in past_dates.iterrows():
            StockPrice.objects.create(stock_id=stock, date=index, close_price=row['Close'])
        
        for index, row in lst_output_df.iterrows():
            StockPrice.objects.create(stock_id=stock, date=row['Date'], close_price=row['Predicted Close'])
        
        stock_prices = StockPrice.objects.filter(stock_id=stock)
        
        serializer = StockPriceSerializer(stock_prices, many=True)
        return Response(serializer.data)
    

''' Usage: "http://api/stockanalysis/AAPL/" '''
# Create a class based view that takes a stock_id and returns the news related to the stock
class StockAnalysisView(APIView):
    def get(self, request, stock_id):
        # try:
        tick = yf.Ticker(stock_id.upper())
        news = tick.news
        
        # For all news, join the title of the news to form a list of titles
        titles = []
        for new in news:
            titles.append(new['title'])    

        print(titles)         
        
        model = TFBertForSequenceClassification.from_pretrained("ahmedrachid/FinancialBERT-Sentiment-Analysis",num_labels=3, from_pt=True)
        tokenizer = BertTokenizer.from_pretrained("ahmedrachid/FinancialBERT-Sentiment-Analysis")

        nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

        sentences = titles 
        results = nlp(sentences)
        print(results)
            

        # except:
        #     return Response("Stock not found", status=404)
        # Create a list to store the results
        sentiment_results = []

        # Loop through the results
        for i in range(len(results)):
            # Append the result to the sentiment_results list
            sentiment_results.append({
            'sentences': sentences[i],
            'label': results[i]['label'],
            'score': results[i]['score']
            })

        # Return the sentiment_results list as a response
        return Response(sentiment_results)
    



