import sys, os
sys.path.append('/home/dblab/nfs/kubeflow-user-example-com-electric-test-pvc-1195405b-000f-484c-9f0d-6c9160dc85a1/workspaces/ePredict/')
import numpy as np
import random
import pandas as pd
from pylab import mpl, plt
from datetime import datetime
import math, time
import itertools
import datetime
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm
from . import trainUtil as ut
import pickle



def main() : 
    pd.set_option('display.max_columns', None)
    # 데이터 불러오기(10분 단위 데이터)
    df1D = pd.read_csv('data/new_total_pv_0831.csv', parse_dates=['updated'], encoding='utf-8', )
    dateColumn = 'updated'
    freq = 'D'

    resultDf = ut.slicerFreq(df1D, freq, dateColumn) #일 단위 데이터로 변환 및 결측치 선형 보간

    trainS = '2019-01-01'
    trainE = '2021-07-31'
    esPatience = 25
    num_epochs = 20

    ARTrainset,y_train_pred ,y_train , loss = ut.Training(num_epochs, resultDf, trainS, trainE,esPatience)
    ut.PRPlot('Training', y_train_pred.detach().numpy(), y_train)
    print("Traning MSE loss : ",loss.item())


    with open("data/ARTrainset.pkl","wb") as fw:
        pickle.dump(ARTrainset, fw)


if __name__ == "__main__":
    main()