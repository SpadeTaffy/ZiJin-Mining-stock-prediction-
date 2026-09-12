import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame as df
import plotly.express as px
import akshare as ak
import yfinance as yf
import numpy as np

####使用Lasso####
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso
def Lasso_test(x_train,y_train,x_test,y_test,Max_iter=10000):
    pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lasso', Lasso(max_iter=Max_iter))
    ])

    ts_split = TimeSeriesSplit(n_splits=5)

    parameters = {
    'lasso__alpha': np.logspace(-5, 1, 1000)
}

    grid_lasso = GridSearchCV(
    estimator=pipe,
    param_grid=parameters,
    cv=ts_split,
    scoring='neg_root_mean_squared_error',
    refit=True
)

    grid_lasso.fit(x_train, y_train)

    Y_pred = grid_lasso.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, Y_pred))
    r2 = r2_score(y_test, Y_pred)

    print("RMSE：", rmse)
    print("R²：", r2)
    plt.figure(figsize=(20,6))
    plt.plot(y_test.index, y_test, label='Actual')
    plt.plot(y_test.index, Y_pred, label='Prediction')
    plt.legend()
    plt.show()

    coef = grid_lasso.best_estimator_.named_steps["lasso"].coef_
    display(pd.Series(coef, index=x_train.columns))
    
    
####计算技术指标####
def calculate_wilder_RSI(df, periods=14):
    # wilder's RSI
    delta = df.diff()
    up, down = delta.copy(), delta.copy()

    up[up < 0] = 0
    down[down > 0] = 0

    rUp = up.ewm(span=periods,adjust=False).mean()
    rDown = down.ewm(span=periods, adjust=False).mean().abs()

    rsi = 100 - 100 / (1 + rUp / rDown)
    return rsi

def calculate_MACD(df, nslow=26, nfast=12):
    emaslow = df.ewm(span=nslow, min_periods=nslow, adjust=True, ignore_na=False).mean()
    emafast = df.ewm(span=nfast, min_periods=nfast, adjust=True, ignore_na=False).mean()
    dif = emafast - emaslow
    Dea = dif.ewm(span=9, min_periods=9, adjust=True, ignore_na=False).mean()
    Histo = dif - Dea
    return dif, Histo