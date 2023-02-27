import tensorflow as tf
import os
import numpy as np
#import product
import itertools

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#config = tf.compat.v1.ConfigProto(device_count = {'GPU': 0})
#sess = tf.compat.v1.Session(config=config)

#optimizer, dropout, batch normalization, regularizatiom
l_dropout = [0, 0.2, 0.5, 0.9]
l_batchnorm = [True, False]
l_reg = [True, False]
l_regtypes = ['l1', 'l2', 'l1_l2']
l_regvalues = [0.01, 0.001]
l_epochs = [5, 10, 25]
l_optimizers = ['adam', 'sgd', 'rmsprop']
l_losses = ['binary_crossentropy', 'categorical_crossentropy', 'hinge', 'mean_squared_error', 'poisson']

#open a file
filepath = "../datasets/mushroom/agaricus-lepiota.data"
file = open(filepath, "r")

#import data 
data = file.read().splitlines()

#close file
file.close()

#split data into features and labels
features = []
labels = []
for line in data:
    line = line.split(",")
    #conevrt all letters to numbers
    for i in range(len(line)-1):
        line[i+1] = ord(line[i+1]) - 97
    #convert labels to 0 or 1
    line[0] = 1 if line[0] == "e" else 0
    features.append(line[1:])
    labels.append(line[0])


#split data into training and testing
training_features = features[:int(len(features)*0.8)]
training_labels = labels[:int(len(labels)*0.8)]
testing_features = features[int(len(features)*0.8):]
testing_labels = labels[int(len(labels)*0.8):]

#convert features and labels to numpy arrays
training_features = np.array(training_features)
training_labels = np.array(training_labels)
testing_features = np.array(testing_features)
testing_labels = np.array(testing_labels)

#open a text file and add the name of the model
file = open("results.csv", "a")
count = 0
loaded = False
prod = itertools.product(l_dropout, l_batchnorm, l_reg, l_regtypes, l_regvalues, l_epochs, l_optimizers, l_losses)
#file.write("Total:"+str(len(list(prod)))+"\n")
file.write("Number, Model, Optimizer, Loss, Epochs, Dropout1, Batch Normalization, Regularization 1, Regularization Type 1, Regularization Value 1, Training Accuracy, Testing Accuracy, Training Loss, Testing Loss\n")
for d1, norm, reg1, regt1, regv1, epoch, opt, lloss in itertools.product(l_dropout, l_batchnorm, l_reg, l_regtypes, l_regvalues, l_epochs, l_optimizers, l_losses):
    print("reg1: ", reg1, " regt1: ", regt1, " regv1: ", regv1, " epoch: ", epoch, " opt: ", opt, " lloss: ", lloss)
    if os.path.exists("models/mushroom_model_"+str(count)):
        model = tf.keras.models.load_model('models/mushroom_model_'+str(count))
        print('Model Loaded'+'models/mushroom_model_'+str(count))
        loaded = True
    else:
        #create a Sequential model
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.BatchNormalization()) if norm else None
        if reg1 is True:
            if regt1 == "l1":
                model.add(tf.keras.layers.Dense(22, activation="relu", input_shape=(22,), kernel_regularizer=tf.keras.regularizers.l1(regv1)))
            elif regt1 == "l2":
                model.add(tf.keras.layers.Dense(22, activation="relu", input_shape=(22,), kernel_regularizer=tf.keras.regularizers.l2(regv1)))
            elif regt1 == "l1_l2":
                model.add(tf.keras.layers.Dense(22, activation="relu", input_shape=(22,), kernel_regularizer=tf.keras.regularizers.l1_l2(regv1)))
        else:
            model.add(tf.keras.layers.Dense(22, activation="relu", input_shape=(22,)))
        model.add(tf.keras.layers.Dropout(d1)) if d1 != 0 else None

        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

        #compile model
        model.compile(optimizer=opt, loss=lloss, metrics=["accuracy"])
    
        print("Training")
        #train model
        #print training accuracy every 10 epochs
        class PrintAccuracy(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs={}):
                if epoch % 10 == 0:
                    print("Training accuracy: ", logs.get("accuracy"))
    
        model.fit(training_features, training_labels, epochs=epoch, callbacks=[PrintAccuracy()])

    #evaluate model
    eval = model.evaluate(testing_features, testing_labels)
    acc = eval[1]
    loss = eval[0]
    eval = model.evaluate(training_features, training_labels)
    test_acc = eval[1]
    test_loss = eval[0]

    
    #save model results to file
    print("Saving")
    print("Model: ", model)
    print("Optimizer: ", opt)
    print("Loss: ", lloss)
    print("Epochs: ", epoch)
    print("Dropout1: ", d1)
    #print("Dropout2: ", d2)
    print("Batch Normalization: ", norm)
    print("Regularization 1: ", reg1)
    print("Regularization Type 1: ", regt1)
    print("Regularization Value 1: ", regv1)
    #print("Regularization 2: ", reg2)
    #print("Regularization Type 2: ", regt2)
    #print("Regularization Value 2: ", regv2)
    print("Training Accuracy: ", test_acc)
    print("Testing Accuracy: ", acc)
    print("Training Loss: ", test_loss)
    print("Testing Loss: ", loss)
    file.write(str(count)+", "+str(model)+", "+str(opt)+", "+str(lloss)+", "+str(epoch)+", "+str(d1)+", "+str(norm)+", "+str(reg1)+", "+str(regt1)+", "+str(regv1)+", "+str(test_acc)+", "+str(acc)+", "+str(test_loss)+", "+str(loss)+"\n")
    file.flush()

    #save model
    if not loaded:
        model.save("models/mushroom_model_"+str(count))
        model.save_weights("weights/mushroom_model_weights_"+str(count))
        print('Saved mushroom_model_'+str(count))
    count = count + 1
    loaded = False

#load model
#model = tf.keras.models.load_model("mushroom_model")
