import warnings
import pandas as pd
import numpy as np
import random
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning) 

#1. 6-position hot encoded feature
capShape = {"b": [1, 0, 0, 0, 0, 0],
            "c": [0, 1, 0, 0, 0, 0],
            "x": [0, 0, 1, 0, 0, 0],
            "f": [0, 0, 0, 1, 0, 0],
            "k": [0, 0, 0, 0, 1, 0],
            "s": [0, 0, 0, 0, 0, 1]
            }
#2. 4-position hot encoded feature
capSurface = {"f": [1, 0, 0, 0],
              "g": [0, 1, 0, 0],
              "y": [0, 0, 1, 0],
              "s": [0, 0, 0, 1]
            }
#3. 10-position hot encoded feature
capColor = {"n": [1, 0, 0, 0, 0, 0, 0, 0, 0 ,0],
            "b": [0, 1, 0, 0, 0, 0, 0, 0, 0 ,0],
            "c": [0, 0, 1, 0, 0, 0, 0, 0, 0 ,0],
            "g": [0, 0, 0, 1, 0, 0, 0, 0, 0 ,0],
            "r": [0, 0, 0, 0, 1, 0, 0, 0, 0 ,0],
            "p": [0, 0, 0, 0, 0, 1, 0, 0, 0 ,0],
            "u": [0, 0, 0, 0, 0, 0, 1, 0, 0 ,0],
            "e": [0, 0, 0, 0, 0, 0, 0, 1, 0 ,0],
            "w": [0, 0, 0, 0, 0, 0, 0, 0, 1 ,0],
            "y": [0, 0, 0, 0, 0, 0, 0, 0, 0 ,1]
            }
#4. 2-position hot encoded feature
bruises = {"t": [1, 0], 
            "f": [0, 1]
            }
#5. 9-position hot encoded feature
odor = {"a": [1, 0, 0, 0, 0, 0, 0, 0, 0],
        "l": [0, 1, 0, 0, 0, 0, 0, 0, 0],
        "c": [0, 0, 1, 0, 0, 0, 0, 0, 0],
        "y": [0, 0, 0, 1, 0, 0, 0, 0, 0],
        "f": [0, 0, 0, 0, 1, 0, 0, 0, 0],
        "m": [0, 0, 0, 0, 0, 1, 0, 0, 0],
        "n": [0, 0, 0, 0, 0, 0, 1, 0, 0],
        "p": [0, 0, 0, 0, 0, 0, 0, 1, 0],
        "s": [0, 0, 0, 0, 0, 0, 0, 0, 1]
        }
#6. 4-position hot encoded feature
gillAttachment = {"a": [1, 0, 0, 0],
                "d": [0, 1, 0, 0],
                "f": [0, 0, 1, 0],
                "n": [0, 0, 0, 1]
                }
#7. 3-position hot encoded feature
gillSpacing = {"c": [1, 0, 0],
                "w": [0, 1, 0],
                "d": [0, 0, 1]
                }
#8. 2-position hot encoded feature
gillSize = {"b": [1, 0],
            "n": [0, 1]
            }
#9. 12-position hot encoded feature
gillColor = {"k": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "n": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "b": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "h": [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            "g": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            "r": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            "o": [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            "p": [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            "u": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            "e": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            "w": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            "y": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
            }
#10. 2-position hot encoded feature
stalkShape = {"e": [1, 0],
            "t": [0, 1]
            }
#11. 7-position hot encoded feature
stalkRoot = {"b": [1, 0, 0, 0, 0, 0, 0],
            "c": [0, 1, 0, 0, 0, 0, 0],
            "u": [0, 0, 1, 0, 0, 0, 0],
            "e": [0, 0, 0, 1, 0, 0, 0],
            "z": [0, 0, 0, 0, 1, 0, 0],
            "r": [0, 0, 0, 0, 0, 1, 0],
            "?": [0, 0, 0, 0, 0, 0, 1]
            }
#12. 4-position hot encoded feature
stalkSurfaceAboveRing = {"f": [1, 0, 0, 0],
                        "y": [0, 1, 0, 0],
                        "k": [0, 0, 1, 0],
                        "s": [0, 0, 0, 1]
                        }
#13. 4-position hot encoded feature
stalkSurfaceBelowRing = {"f": [1, 0, 0, 0],
                        "y": [0, 1, 0, 0],
                        "k": [0, 0, 1, 0],
                        "s": [0, 0, 0, 1]
                        }
#14. 9-position hot encoded feature
stalkColorAboveRing = {"n": [1, 0, 0, 0, 0, 0, 0, 0, 0],
                        "b": [0, 1, 0, 0, 0, 0, 0, 0, 0],
                        "c": [0, 0, 1, 0, 0, 0, 0, 0, 0],
                        "g": [0, 0, 0, 1, 0, 0, 0, 0, 0],
                        "o": [0, 0, 0, 0, 1, 0, 0, 0, 0],
                        "p": [0, 0, 0, 0, 0, 1, 0, 0, 0],
                        "e": [0, 0, 0, 0, 0, 0, 1, 0, 0],
                        "w": [0, 0, 0, 0, 0, 0, 0, 1, 0],
                        "y": [0, 0, 0, 0, 0, 0, 0, 0, 1]
                        }
#15. 9-position hot encoded feature
stalkColorBelowRing = {"n": [1, 0, 0, 0, 0, 0, 0, 0, 0],
                        "b": [0, 1, 0, 0, 0, 0, 0, 0, 0],
                        "c": [0, 0, 1, 0, 0, 0, 0, 0, 0],
                        "g": [0, 0, 0, 1, 0, 0, 0, 0, 0],
                        "o": [0, 0, 0, 0, 1, 0, 0, 0, 0],
                        "p": [0, 0, 0, 0, 0, 1, 0, 0, 0],
                        "e": [0, 0, 0, 0, 0, 0, 1, 0, 0],
                        "w": [0, 0, 0, 0, 0, 0, 0, 1, 0],
                        "y": [0, 0, 0, 0, 0, 0, 0, 0, 1]
                        }
#16. 2-position hot encoded feature
veilType = {"p": [1, 0],
            "u": [0, 1]
            }
#17. 4-position hot encoded feature
veilColor = {"n": [1, 0, 0, 0],
            "o": [0, 1, 0, 0],
            "w": [0, 0, 1, 0],
            "y": [0, 0, 0, 1]
            }
#18. 3-position hot encoded feature
ringNumber = {"n": [1, 0, 0],
            "o": [0, 1, 0],
            "t": [0, 0, 1]
            }
#19. 8-position hot encoded feature
ringType = {"c": [1, 0, 0, 0, 0, 0, 0, 0],
            "e": [0, 1, 0, 0, 0, 0, 0, 0],
            "f": [0, 0, 1, 0, 0, 0, 0, 0],
            "l": [0, 0, 0, 1, 0, 0, 0, 0],
            "n": [0, 0, 0, 0, 1, 0, 0, 0],
            "p": [0, 0, 0, 0, 0, 1, 0, 0],
            "s": [0, 0, 0, 0, 0, 0, 1, 0],
            "z": [0, 0, 0, 0, 0, 0, 0, 1]
            }
#20. 9-position hot encoded feature
sporePrintColor = {"k": [1, 0, 0, 0, 0, 0, 0, 0, 0],
                "n": [0, 1, 0, 0, 0, 0, 0, 0, 0],
                "b": [0, 0, 1, 0, 0, 0, 0, 0, 0],
                "h": [0, 0, 0, 1, 0, 0, 0, 0, 0],
                "r": [0, 0, 0, 0, 1, 0, 0, 0, 0],
                "o": [0, 0, 0, 0, 0, 1, 0, 0, 0],
                "u": [0, 0, 0, 0, 0, 0, 1, 0, 0],
                "w": [0, 0, 0, 0, 0, 0, 0, 1, 0],
                "y": [0, 0, 0, 0, 0, 0, 0, 0, 1]
                }
#21. 6-position hot encoded feature
population = {"a": [1, 0, 0, 0, 0, 0],
            "c": [0, 1, 0, 0, 0, 0],
            "n": [0, 0, 1, 0, 0, 0],
            "s": [0, 0, 0, 1, 0, 0],
            "v": [0, 0, 0, 0, 1, 0],
            "y": [0, 0, 0, 0, 0, 1],
            }
#22. 7-position hot encoded feature
habitat = {"g": [1, 0, 0, 0, 0, 0, 0],
            "l": [0, 1, 0, 0, 0, 0, 0],
            "m": [0, 0, 1, 0, 0, 0, 0],
            "p": [0, 0, 0, 1, 0, 0, 0],
            "u": [0, 0, 0, 0, 1, 0, 0],
            "w": [0, 0, 0, 0, 0, 1, 0],
            "d": [0, 0, 0, 0, 0, 0, 1],
            }

#0. 1-position hot encoded output
edibleOrPoisonous = {"e": [0], "p": [1]}

df = pd.read_csv("../../datasets/mushroom/agaricus-lepiota.data", header = None)

fullData = {}
fullDataMatrix = []
fullIndexMatrix = []

for i in range(0, 8124):
    instanceMatrix = [[i],
                    edibleOrPoisonous[df.loc[i][0]],
                    capShape[df.loc[i][1]],
                    capSurface[df.loc[i][2]],
                    capColor[df.loc[i][3]],
                    bruises[df.loc[i][4]],
                    odor[df.loc[i][5]],
                    gillAttachment[df.loc[i][6]],
                    gillSpacing[df.loc[i][7]],
                    gillSize[df.loc[i][8]],
                    gillColor[df.loc[i][9]],
                    stalkShape[df.loc[i][10]],
                    stalkRoot[df.loc[i][11]],
                    stalkSurfaceAboveRing[df.loc[i][12]],
                    stalkSurfaceBelowRing[df.loc[i][13]],
                    stalkColorAboveRing[df.loc[i][14]],
                    stalkColorBelowRing[df.loc[i][15]],
                    veilType[df.loc[i][16]],
                    veilColor[df.loc[i][17]],
                    ringNumber[df.loc[i][18]],
                    ringType[df.loc[i][19]],
                    sporePrintColor[df.loc[i][20]],
                    population[df.loc[i][21]],
                    habitat[df.loc[i][22]]]

    fullData[str(i)] = instanceMatrix
    fullDataMatrix.append(instanceMatrix)
    fullIndexMatrix.append(i)

#5416 is 2/3 of 8124, 2708 is 1/3, 1354 is 1/6, 677 is 1/12
sizeOfTrainingSet = [5416, 2708, 1354, 677, 500, 400, 300, 200, 100, 50, 25]
accuracyReadings = ["5416 - Two Thirds", "2708 - One Third", "1354 - One Sixth", "677 - One Twelfth", "500", "400", "300", "200", "100", "50", "25"]
sizeIterator = 0
hiddenLayers = [10, 8, 6, 4, 2]
hiddenLayerNumbers = ["HL 10", "HL 8", "HL 6", "HL 4", "HL 2"]
maximumIteration = [1000, 500, 250, 125]
maxiumIterationNumber = ["MI 1000", "MI 500", "MI 250", "MI 125"]
activation_layer = ["relu", "tanh", "logistic", "identity"]
optimization_layer = ["sgd", "adam"]
leanring_rate = [0.001, 0.0001, 0.01]

dataGatherMainRun = {}
dataGatherLearningRates = {}

for k in sizeOfTrainingSet:
    if k not in dataGatherMainRun:
        dataGatherMainRun[k] = {}
    if k not in dataGatherLearningRates:
        dataGatherLearningRates[k] = {}
    
    selection = np.random.randint(8124, size = k)
    selectionList = list(selection)
    trainingSetMatrix = []
    trainingSetMatrixANSWERS = []
    trainingSetMatrixDATA = []
    TESTTraining_TestingSetANSWERS = []
    TESTTraining_TestingSetDATA = []
    iteratorOne = 0
    for i in selectionList:
        trainingSetMatrix.append(fullData[str(i)])
        trainingSetMatrixANSWERS.append(trainingSetMatrix[iteratorOne][1][0])
        concatenatedONEHotEncodings = []
        for j in range(0, 22):
            concatenatedONEHotEncodings = np.concatenate((concatenatedONEHotEncodings, np.array(trainingSetMatrix[iteratorOne][2 + j])))
        trainingSetMatrixDATA.append(list(concatenatedONEHotEncodings))
        iteratorOne += 1
    TESTIndexesTraining = set(fullIndexMatrix) - set(selectionList)
    TESTIndexesTrainingList = list(TESTIndexesTraining)
    random.shuffle(TESTIndexesTrainingList)
    TESTTraining_TestingSet = []
    iteratorTwo = 0
    for i in TESTIndexesTrainingList:
        TESTTraining_TestingSet.append(fullData[str(i)])
        TESTTraining_TestingSetANSWERS.append(TESTTraining_TestingSet[iteratorTwo][1][0])
        concatenatedONEHotEncodings = []
        for j in range(0, 22):
            concatenatedONEHotEncodings = np.concatenate((concatenatedONEHotEncodings, np.array(TESTTraining_TestingSet[iteratorTwo][2 + j])))
        TESTTraining_TestingSetDATA.append(list(concatenatedONEHotEncodings))
        iteratorTwo += 1
        
    optimizationRun = 0
    for l in optimization_layer:
        if l not in dataGatherMainRun[k]:
            dataGatherMainRun[k][l] = {}
        if l not in dataGatherLearningRates[k]:
            dataGatherLearningRates[k][l] = {}
        activationRun = 0
        for h in activation_layer:
            if h not in dataGatherMainRun[k][l]:
                dataGatherMainRun[k][l][h] = {}
            if h not in  dataGatherLearningRates[k][l]:
                dataGatherLearningRates[k][l][h] = {}
            indexHL = 0
            for m in hiddenLayers:
                if m not in dataGatherMainRun[k][l][h]:
                    dataGatherMainRun[k][l][h][m] = {}
                if m not in dataGatherLearningRates[k][l][h]:
                    dataGatherLearningRates[k][l][h][m] = {}
                maxIterationIndex = 0
                for n in maximumIteration:
                    if n not in dataGatherLearningRates[k][l][h][m]:
                        dataGatherLearningRates[k][l][h][m][n] = {}
                    if k <= 500:
                        pIndex = 0
                        for p in leanring_rate:
                            mlp = MLPClassifier(hidden_layer_sizes=(m,), activation = h, solver = l, max_iter = n, learning_rate_init = p)
                        
                            mlp.fit(trainingSetMatrixDATA, trainingSetMatrixANSWERS)
            
                            testOnTraining = mlp.predict(TESTTraining_TestingSetDATA)

                            accuracy = accuracy_score(TESTTraining_TestingSetANSWERS, testOnTraining)
                            
                            print("ACCURACY, total size " + accuracyReadings[sizeIterator] + ", " + l + ", " + h + ", " + hiddenLayerNumbers[indexHL] + ", " +  maxiumIterationNumber[maxIterationIndex] + ", " + "learning rate " + str(p) + ": " +  str(accuracy))
                            dataGatherLearningRates[k][l][h][m][n][p] = {accuracy}
                            pIndex += 1
                    else:
                        mlp = MLPClassifier(hidden_layer_sizes=(m,), activation = h, solver = l, max_iter = n)
                        
                        mlp.fit(trainingSetMatrixDATA, trainingSetMatrixANSWERS)
            
                        testOnTraining = mlp.predict(TESTTraining_TestingSetDATA)

                        accuracy = accuracy_score(TESTTraining_TestingSetANSWERS, testOnTraining)
                        
                        dataGatherMainRun[k][l][h][m][n] = {accuracy}

                        print("ACCURACY, total size " + accuracyReadings[sizeIterator] + ", " + l + ", " + h + ", " + hiddenLayerNumbers[indexHL] + ", " +  maxiumIterationNumber[maxIterationIndex] + ": " +  str(accuracy))
                    maxIterationIndex += 1
                indexHL += 1
            activationRun += 1
        optimizationRun += 0
    sizeIterator += 1
    
dtMainRun = pd.DataFrame.from_dict(dataGatherMainRun)
dtLearningRate = pd.DataFrame.from_dict(dataGatherLearningRates)

dtMainRun.to_csv("MushroomNNMainRunOutput.csv", index = False)
dtLearningRate.to_csv("MushroomNNLearnignRatesOutput.csv", index = False)

