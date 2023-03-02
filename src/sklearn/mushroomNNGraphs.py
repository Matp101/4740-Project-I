import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ast

df1M = pd.read_csv("MushroomNNMainRunOutput.csv")
df2LR = pd.read_csv("MushroomNNLearnignRatesOutput.csv")
df1M_dict = df1M.to_dict(orient = "index")
df2LR_dict = df2LR.to_dict(orient = "index")

sizeOfTrainingSet = [5416, 2708, 1354, 677, 500, 400, 300, 200, 100, 50, 25]
limitedTrainingToUpper = [5416, 2708, 1354, 677]
limitedTrainingToLower = [500, 400, 300, 200, 100, 50, 25]
hiddenLayers = [10, 8, 6, 4, 2]
maximumIteration = [125, 250, 500, 1000]
activation_layer = ["relu", "tanh", "logistic", "identity"]
optimization_layer = ["sgd", "adam"]
# 0 for sgd, 1 for adam
opt_L_numberCode = [0, 1]
learning_rate = [0.001, 0.0001, 0.01]

for i in opt_L_numberCode:
    for t in sizeOfTrainingSet:
        df1M_dict[i][str(t)] = ast.literal_eval(df1M_dict[i][str(t)])
        df2LR_dict[i][str(t)] = ast.literal_eval(df2LR_dict[i][str(t)])
        
fig, axs = plt.subplots(1, 2, subplot_kw={'projection': '3d'})
fig.subplots_adjust(hspace=0.6, wspace=0.6)

color_dict = {"5416": "m", "2708": "b", "1354": "y", "677": "r"}
shape_dict = {"relu": "o", "tanh": "^", "logistic": "D", "identity": "s"}

for s in limitedTrainingToUpper:
    for a in activation_layer:
        for h in hiddenLayers:
            miIndex = 0
            for m in maximumIteration:
                x1 = list(df1M_dict[0][str(s)][a][h])[miIndex]
                x2 = list(df1M_dict[1][str(s)][a][h])[miIndex]
                xSGD = x1
                xADAM = x2
                ySGD = m
                yADAM = m
                zSGD = h
                zADAM = h
                color = color_dict[str(s)]
                shape = shape_dict[a]
                
                axs[0].scatter(xSGD, ySGD, zSGD, c = color, marker = shape)
                axs[1].scatter(xADAM, yADAM, zADAM, c = color, marker = shape)
                
                miIndex += 1
                
#'SGD Accuracy plot with varying total batch size and optimization layer'         
axs[0].set_xlabel('Accuracy')
axs[0].set_ylabel('Max Iter')
axs[0].set_zlabel('HL Num')
axs[0].set_title('SGD')

#'ADAM Accuracy plot with varying total batch size and optimization layer'
axs[1].set_xlabel('Accuracy')
axs[1].set_ylabel('Max Iter')
axs[1].set_zlabel('HL Num')
axs[1].set_title('ADAM')

plt.show()