import  pandas as pd
import numpy as np
import requests
import lightgbm as lgb
import sklearn
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.model_selection import ExpandingWindowSplitter, ForecastingGridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import  mean_absolute_percentage_error



###### another Lgb model using skTime model selection rather than manuel parameter, seleciton


def to_df(data_loader):
    xdd = data_loader
    x = pd.DataFrame(xdd)
    print(x)

    ydd = data_loader.close
    y = pd.Series(ydd)
    print(y)
    return x,y
magicGlobe = 3
f = requests.get("https://min-api.cryptocompare.com/data/v2/histoday?fsym=BTC&tsym=USD&limit=2000").json()['Data']['Data']
g = pd.DataFrame(f)
df = g[['high', 'low', 'volumeto', 'close']]






x,y = to_df(df)
xTrain,xTest,yTrain,yTest = train_test_split(x,y, test_size=0.2, random_state=None, shuffle=False)
print(yTrain)
yTrain,yTest = pd.Series(yTrain),  pd.Series(yTest)
print(yTest)


def genModel():
    regressor = lgb.LGBMRegressor()
    forecaster = make_reduction(regressor, window_length=5, strategy="recursive")
    return forecaster

def getErrors(series_test, forecast):

    mae = mean_absolute_error(series_test, forecast)
    mape = mean_absolute_percentage_error(series_test, forecast)
    return mae,mape

def  gridSearchPredict(xtrain,ytrain,ytest,xtest,forecaster,paramGrid):
    cv = ExpandingWindowSplitter(initial_window=int(len(ytrain)*0.65))
    gscv = ForecastingGridSearchCV(
        forecaster,strategy="refit",cv=cv, param_grid=paramGrid
    )
    gscv.fit(y=ytrain,X=xtrain)
    print(f"Recommended Params: {gscv.best_params_}")
    fh=np.arange(len(ytest))+1
    print(fh)
    yPred = gscv.predict(fh=fh)
    yPred = np.array(yPred)
    ytest = np.array(ytest)
    print(yPred)
    mae,mape = getErrors(ytest,yPred)
    return mae,mape,yPred

paramGrid = {"window_length": [5,10,15,20,25,30]}
forecaster= genModel()
finalMae, finalMape, predd = gridSearchPredict(xTrain,yTrain,yTest,xTest,forecaster,paramGrid) # try  both x and y data  



#reg = lgb.LGBMRegressor()
#reg.fit(xTrain,yTrain)
#predd = reg.predict(xTest)
print(finalMae)
print(predd)
scre = r2_score(yTest, predd)
print(scre)
