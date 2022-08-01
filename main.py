import  pandas as pd
import numpy as np
import requests
import lightgbm as lgb
import sklearn
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


def to_df(data_loader):
    xdd = data_loader
    x = pd.DataFrame(xdd)
    print(x)

    ydd = data_loader.close
    y = pd.Series(ydd, name='trgt')
    print(y)
    return x,y
magicGlobe = 3
f = requests.get("https://min-api.cryptocompare.com/data/v2/histohour?fsym=BTC&tsym=USD&limit=2000").json()['Data']['Data']
g = pd.DataFrame(f)
df = g[['high', 'low', 'volumeto', 'close']]





#preprocessing
x,y = to_df(df)
xTrain,xTest,yTrain,yTest = train_test_split(x,y, test_size=0.2, shuffle=False)
print(yTrain)
yTrain,yTest = pd.Series(yTrain),  pd.Series(yTest)
print(yTest)
lgbTrain = lgb.Dataset(xTrain,yTrain)
lgbEval = lgb.Dataset(xTest,yTest, reference=lgbTrain)
params = {
    'task': 'train',
    'boosting': 'gbdt',
    'objective': 'regression',
    'num_leaves': 10,
    'learning_rate': 0.05,
    'metric': {'l2','l1'},
    'verbose':  -1
}
#fit and predict
model = lgb.train(params, train_set=lgbTrain, valid_sets=lgbEval)
model.predict(xTest)
print(xTest)
