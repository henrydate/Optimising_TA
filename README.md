![](Trading%20Wizard.jpg)
# <div align = "center"> Challenge-Two
## <div align = "center"> Team Two
### <div align = "center"> Akhil Kavuri, Brendan Van Maanen, Danny Milsom, Henry Date and Ling Dong
#### <div align = "center"> *AN INTERACTIVE TRADING & INVESTEMENT MACHINE BOT*

## 1. Project Aim:   
To build an interactive and intuitive trading BOT that optimises techical indicators through machine learning (Ensemble method) to provide highly accurate investment and trading recommendations. Importantly the BOT is to reliablely predict how price interacts with technical indicators (EMA) in order to maximise indicator trading performance.   

The current version focuses on the SP500, Nasdaq 100, Russel 200, Dow Jones indices and ASX200. However the model can be for individual stocks - subject to data availability. 
	
A key feature of the BOT is to allow for a high degree of "fine tuning" through the Ensemble method. 
   
## 2. Key Technologies:
  
To use machine learning through the Ensemble Method to filter accurate buy/sell recommendations from four key technical indicators.


   ### 2.1 [MACD](https://investopedia.com/terms/m/macd.asp)  
   Moving average convergence/divergence (MACD, or MAC-D) is a trend-following momentum indicator that shows the relationship between two exponential moving averages      (EMAs) of a security’s price. The MACD line is calculated by subtracting the 26-period EMA from the 12-period EMA.
   
   ### 2.2 [Bollinger Bands](https://www.investopedia.com/terms/b/bollingerbands.asp)  
  A Bollinger Band® is a technical analysis tool defined by a set of trendlines plotted two standard deviations (positively and negatively) away from a simple moving average (SMA) of a security's price, but which can be adjusted to user preferences.
  
  ### 2.3 [Exponential Moving Averages](https://investopedia.com/terms/e/ema.asp)  
   An exponential moving average (EMA) is a type of moving average (MA) that places a greater weight and significance on the most recent data points. The exponential moving average is also referred to as the exponentially weighted moving average. An exponentially weighted moving average reacts more significantly to recent price changes than a simple moving average simple moving average (SMA), which applies an equal weight to all observations in the period.
  
  ### 2.4 [Relative Strength Index](https://investopedia.com/terms/r/rsi.asp)  
  The relative strength index (RSI) is a momentum indicator used in technical analysis. RSI measures the speed and magnitude of a security's recent price changes to evaluate overvalued or undervalued conditions in the price of that security.

[Investopedia](https://www.investopedia.com)	
  
  ### 2.5 [Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html)  
  The goal of ensemble methods is to combine the predictions of several base estimators built with a given learning algorithm in order to improve generalizability/  robustness over a single estimator.  
  Ensemble methods use:
  
  #### 2.5.1 [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logistic+regression)  
  In the multiclass case, the training algorithm uses the one-vs-rest (OvR) scheme if the ‘multi_class’ option is set to ‘ovr’, and uses the cross-entropy loss if the ‘multi_class’ option is set to ‘multinomial’. (Currently the ‘multinomial’ option is supported only by the ‘lbfgs’, ‘sag’, ‘saga’ and ‘newton-cg’ 	solvers.)  
	
  #### 2.5.2 [GaussianNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html?highlight=gaussiannb#sklearn.naive_bayes.GaussianNB)         
  Can perform online updates to model parameters via partial_fit. For details on algorithm used to update feature means and variance online, see Stanford CS tech         report STAN-CS-79-773 by Chan, Golub, and LeVeque. 
	
  #### 2.5.3 [XGBClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)    
  This algorithm builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions. In each stage       n_classes_ regression trees are fit on the negative gradient of the loss function, e.g. binary or multiclass log loss. Binary classification is a special case         where only a single regression tree is induced.  
	
  #### 2.5.4 [Voting Classifier](https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier)  
  The idea behind the VotingClassifier is to combine conceptually different machine learning classifiers and use a majority vote or the average predicted                 probabilities (soft vote) to predict the class labels. Such a classifier can be useful for a set of equally well performing models in order to balance out             their individual weaknesses.
    
  ### 2.6 [Amazon LEX](https://aws.amazon.com/lex/)  
  Easily add AI that understands intent, maintains context, and automates simple tasks across many languages.
 
    

  ## 3. Key Technology Installations: 
  
  ### 3.1 Importing required libraries
  
      Installing Yahoo Finance  
      !pip install yfinance  
      !pip install pandas_datareader  
      !pip install scikeras  
      !pip install imbalanced-learn  
      !pip install xgboost  
   
  ### 3.2 Creating the feature variables
  
     'def calculated_features(df):
        df['aboveEMA50'] = np.where(df['Close'] > df['EMA50'], 1, 0)
        df['aboveEMA100'] = np.where(df['Close'] > df['EMA100'], 1, 0)
        df['aboveupperBB'] = np.where(df['Close'] > df['upperBB'], 1, 0)
        df['belowlowerBB'] = np.where(df['Close'] < df['lowerBB'], 1, 0)
        df['oversoldRSI'] = np.where(df['nor_RSI'] < 0.30, 1, 0)
        df['overboughtRSI'] = np.where(df['nor_RSI'] > 0.70, 1, 0)
        return df'
      
  ### 3.3 Creating Ensemble   
  
     from sklearn.metrics import log_loss  
     clf1 = LogisticRegression(random_state=1)  
     #clf2 = RandomForestClassifier(n_estimators=50, random_state=1)  
     clf3 = GaussianNB()  
     clf2 = XGBClassifier()  
     eclf = VotingClassifier(estimators=[('lr', clf1), ('xgb', clf2), ('gnb', clf3)],voting='hard')  

     eclf.fit(X_train, y_train)
 
     # predicting the output on the test dataset
     pred_final = eclf.predict(X_test)
 
     # printing log loss between actual and predicted value
     print("The accuracy of the model in percentage is",(accuracy_score(y_test, pred_final)*100))  
    
     The accuracy of the model in percentage is 85.42445274959958  
  
  ### 3.4 Normalising the data using Standard Scaler
  
     # Create a StandardScaler instance
     scaler = StandardScaler()

     # Fit the scaler to the features training dataset
     X_scaler = scaler.fit(X_train)

     # Scale both the training and testing data from the features dataset
     X_train_scaled = X_scaler.transform(X_train)
     X_test_scaled = X_scaler.transform(X_test)

     # encoding class labels as integers
     encoder = LabelEncoder()
     encoder.fit(y_train)
     encoded_Y = encoder.transform(y_train)
  
  ### 3.5 Adding Layers to Neural Network
  
      'def model_demo():
      	classifier_1 = Sequential()
  		classifier_1.add(Dense(units=10, input_dim=20, kernel_initializer='uniform', activation='relu'))
    	classifier_1.add(Dense(units=5, kernel_initializer='uniform', activation='relu'))
    	classifier_1.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    	classifier_1.compile(optimizer='adamax', loss='binary_crossentropy', metrics=['accuracy'])
    	return classifier_1
	model = model_demo()
	model.fit(X_train,y_train, batch_size=20 , epochs=100, verbose=1,shuffle =True)
	predicted_y = model.predict(X_test)'
  
  ### 3.6 Creating Ensemble   
  
      from sklearn.metrics import log_loss  
      clf1 = LogisticRegression(random_state=1)  
      #clf2 = RandomForestClassifier(n_estimators=50, random_state=1)  
      clf3 = GaussianNB()  
      clf2 = XGBClassifier()  
      eclf = VotingClassifier(estimators=[('lr', clf1), ('xgb', clf2), ('gnb', clf3)],voting='hard')  

      eclf.fit(X_train, y_train)
 
     # predicting the output on the test dataset
     pred_final = eclf.predict(X_test)
 
     # printing log loss between actual and predicted value
     print("The accuracy of the model in percentage is",(accuracy_score(y_test, pred_final)*100))  
    
     The accuracy of the model in percentage is 85.42445274959958  
  
  ## 4. Key Input\Output Examples:
	
 ### 4.1 
	INPUT
	for clf, label in zip([clf1, clf2, clf3,eclf], ['Logistic Regression', 'Random Forest', 'Naive Bayes', 'Ensemble']):
    	    scores = cross_val_score(clf, X_test, y_test, scoring='accuracy', cv=5)
            print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
	
	OUTPUT
	Accuracy: 0.45 (+/- 0.00) [Logistic Regression]
        Accuracy: 1.00 (+/- 0.00) [Random Forest]
        Accuracy: 0.50 (+/- 0.02) [Naive Bayes]
        Accuracy: 0.85 (+/- 0.01) [Ensemble]
	
### 4.2
	INPUT
	from sklearn.metrics import classification_report
	testing_signal_predictions = eclf.predict(X_test)
	 # Evaluate the model's ability to predict the trading signal for the testing data
	ensemble_classification_report = classification_report(y_test, testing_signal_predictions)
	print(ensemble_classification_report)
	
	OUTPUT      
	precision    recall  f1-score   support

           0       0.75      1.00      0.86       839
           1       1.00      0.89      0.94       853
           2       0.00      0.00      0.00       181

   	 accuracy                           0.85      1873
  	 macro avg       0.58      0.63      0.60      1873
	weighted avg       0.79      0.85      0.81      1873
### 4.3 
	INPUT
 	# Create a new empty predictions DataFrame using code provided below.
	predictions_df = pd.DataFrame(index=X_test.index)
	predictions_df["predicted_signal"] = testing_signal_predictions
	predictions_df["actual_returns"] = df["Close"].pct_change()
	predictions_df["trading_algorithm_returns"] = predictions_df["actual_returns"] * predictions_df["predicted_signal"]
	predictions_df.head()
	
	OUTPUT
	predicted_signal	actual_returns	trading_algorithm_returns
	Date			
	2001-11-09	0	0.001066	0.000000
	2019-06-13	1	0.004126	0.004126
	2011-01-14	1	0.007245	0.007245
	2020-09-24	0	0.002665	0.000000
	1999-07-09	1	0.005929	0.005929

### 4.4
	INPUT
	import hvplot.pandas
	(((1 + predictions_df[["actual_returns"]]).cumprod()).hvplot(label="Actual Returns", title = ('Cumulative Product Returns of Actual vs Trading Algorithm 	Returns'))) * (((1 + predictions_df[["trading_algorithm_returns"]]).cumprod()).hvplot(label="Trading Algorithm Returns"))
	
#### 4.5
	INPUT
	def predict_timeseries(df):
    for i in range(len(df)):
        prediction = eclf.predict(X_test)
        #####print('prediction', prediction)
        model_df['Buy'][i] = prediction
    print(df.head())    
        
	OUTPUT
    return df
	
	
	'High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close', 'EMA50',
       		'EMA100', 'upperBB', 'middleBB', 'lowerBB', 'RSI', 'nor_RSI',
      		'aboveEMA50', 'aboveEMA100', 'aboveupperBB', 'belowlowerBB',
       		'oversoldRSI', 'overboughtRSI', 'MACD', 'norm_MACD'
		

   #### 4.6 Actual vs Trading Algorithm Returns
  
  ![](https://github.com/Danny-M108/Challenge-Two-/blob/main/Actual_vs_model_cumprod_of_returns.png)

 #### 4.7 Cumulative Product Returns of Actual vs Trading Algorithm Returns
	
![](https://github.com/Danny-M108/Challenge-Two-/blob/main/actual_vs_model_cumprod_of_returns_spy_daily_yf.png)

 #### 4.8 Buy Triggers

![](https://github.com/Danny-M108/Challenge-Two-/blob/main/chart_buy_SPY-daily_yf.png)	

 #### 4.9 Importance of Columns in Feature Dataframe

![](https://github.com/Danny-M108/Challenge-Two-/blob/main/feature_selection.png)
	
## 5 Future Developments and Enhancements:

Future developments and enhancements to include:

1. Risk and Return measurement and analysis through tools such as the Sharpe and Sortino ratios. 

2. Profit and trading outcome comparison against indice performance. For example did the model outperform a simple 12 position (long or short) on the SP500. 

3. Further technical indicators.

4. Further fine tuning of the Ensemble code.
	
  















  
  
  


  
  






  
 



















