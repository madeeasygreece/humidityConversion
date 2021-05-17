# make predictions
from pandas import read_csv
from sklearn.metrics import classification_report

from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.metrics

from matplotlib import pyplot
import numpy as np
import pickle as pckl


##############################################################################
def scaleData(flag, idata, odata):
    if flag == 1:
        scaler = MinMaxScaler()
        nidata = scaler.fit_transform(idata)
        nodata = scaler.fit_transform(odata)
    else:
        nidata = idata
        nodata = odata

    return nidata, nodata


##############################################################################
def readMLP(filename):
    filename="MLP_model.sav"
    loaded_regr=pckl.load(open(filename,"rb"))
    return loaded_regr


##############################################################################
def train_saveMLP(indata, outdata):
    start = 0
    len = 10600

    # regr = MLPRegressor(hidden_layer_sizes=[10, 8], solver='adam', learning_rate='adaptive', verbose=True,
    #                     random_state=1, max_iter=30000).fit(indata[start:len, :], (outdata[start:len, 0:5]))
    regr = MLPRegressor(hidden_layer_sizes=[10, 8], solver='adam', learning_rate='adaptive', verbose=True,
                        random_state=1, max_iter=30000).fit(indata[:, :], (outdata[:, 0:5]))
    print("MP training completed!")
    print("saving the network....")
    filename = "MLP_model.sav"
    pckl.dump(regr, open(filename, "wb"))
    print("saving done")



##############################################################################
#start main program
#read dataset
# dataset = read_csv("./data/humidity_mapping/dataset_git.csv", skiprows=0)
dataset = read_csv("./dataset_git.csv", skiprows=0)

# Split-out validation dataset
d_array = dataset.values

#use columns 0,1,2 as network input
orX = d_array[:,0:3]

#use columns 3,4,5,6,7 as network output
orY = d_array[:,3:8]


#this is how to plot data before scaling
# pyplot.plot(orX[:, 0]) #first column of the table
# pyplot.plot(orX[:, 1]) #second column of the table
# pyplot.show()

#use data scaler or not
scaler_use = 0
if scaler_use == 1:
    X, Y = scaleData(scaler_use, orX, orY)
    # print data after scaling
    pyplot.plot(X[:, 0])  # first column of the table
    pyplot.plot(X[:, 1])  # second column of the table
    pyplot.show()
else:
    X=orX
    Y=orY

#split data into train and test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=1)
in_train = X_train

#train the Neural Netwok and save it to the disk
#this can be ommitted if the network is aleady trained and saved
train_saveMLP(X_train, Y_train)

#read saved network from disk
regr=readMLP("MLP_model.sav")

#make predictions for new, unknow (not learned) inputs X_test is a 2D array
predictedY = regr.predict(X_test)


# It assumes ‘uniform_average’ : scores of all outputs averaged with uniform weight.
RTWO=sklearn.metrics.r2_score(predictedY, Y_test)
print("Average Error1:", RTWO)

# `variance weighted': scores of all outputs are averaged, weighted by the variances of each individual output.
RTWO=sklearn.metrics.r2_score(predictedY, Y_test,multioutput='variance_weighted')
print("Average Error2:", RTWO)

#set plotting range (start/end values)
pls=100
ple=200

pyplot.plot(Y_test[pls:ple, 0], 'b')
pyplot.plot(predictedY[pls:ple,0], 'r')
pyplot.show()


pyplot.plot(Y_test[pls:ple, 1], 'b')
pyplot.plot(predictedY[pls:ple, 1], 'r')
pyplot.show()


pyplot.plot(Y_test[pls:ple, 2], 'b')
pyplot.plot(predictedY[pls:ple, 2], 'r')
pyplot.show()


pyplot.plot(Y_test[pls:ple, 3], 'b')
pyplot.plot(predictedY[pls:ple, 3], 'r')
pyplot.show()


pyplot.plot(Y_test[pls:ple, 4], 'b')
pyplot.plot(predictedY[pls:ple,4], 'r')
pyplot.show()

#Last note:
#if you want to predict the result of a single 1D input like the bellow
oned_in=[24.65, 31.63, 41.28]

# you must first convert it to 2D using [oned_in]
# and you are ready to go
twod_res = regr.predict([oned_in])
print("result",twod_res)


