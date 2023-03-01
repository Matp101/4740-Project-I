import tensorflow as tf
import os
import numpy as np
import itertools
import pandas as pd
from sklearn.model_selection import train_test_split

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# config = tf.compat.v1.ConfigProto(device_count = {'GPU': 0})
# sess = tf.compat.v1.Session(config=config)

# optimizer, dropout, batch normalization, regularizatiom
l_dropout = [0, 0.2, 0.5, 0.9]
l_batchnorm = [True, False]
l_reg = [True, False]
l_regtypes = {'l1': tf.keras.regularizers.l1,
              'l2': tf.keras.regularizers.l2,
              'l1_l2': tf.keras.regularizers.l1_l2}
l_regvalues = [0.01, 0.001]
l_epochs = [5, 10, 25]
l_optimizers = ['adam', 'sgd', 'rmsprop']
l_losses = ['binary_crossentropy', 'categorical_crossentropy',
            'hinge', 'mean_squared_error', 'poisson']

data = pd.read_csv("../datasets/mushroom/agaricus-lepiota.data", header=None)
# make sure to convert all single-char strings to ints
features = data.iloc[:, 1:].applymap(lambda x: ord(x))
labels = data.iloc[:, 0].apply(lambda x: 1 if x == 'e' else 0)

# split data into training and testing
training_features, testing_features, training_labels, testing_labels = train_test_split(
    features, labels, test_size=0.2)

# open a text file and add the name of the model
with open("results.csv", "w") as f:
    count = 0
    loaded = False
    f.write("Number, Model, Optimizer, Loss, Epochs, Dropout1, Batch Normalization, Regularization 1, Regularization Type 1, Regularization Value 1, Training Accuracy, Testing Accuracy, Training Loss, Testing Loss\n")
    for d1, norm, reg1, regt1, regv1, epoch, opt, lloss \
            in itertools.product(l_dropout, l_batchnorm, l_reg, l_regtypes.keys(), l_regvalues, l_epochs, l_optimizers, l_losses):
        if os.path.exists("models/mushroom_model_"+str(count)):
            model = tf.keras.models.load_model('models/mushroom_model_'+str(count))
            print('Model Loaded'+'models/mushroom_model_'+str(count))
            loaded = True
        else:
            print("reg1: ", reg1, " regt1: ", regt1, " regv1: ", regv1,
                " epoch: ", epoch, " opt: ", opt, " lloss: ", lloss)

            # create a Sequential model
            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.BatchNormalization()) if norm else None
            if reg1:
                model.add(tf.keras.layers.Dense(22, activation="relu", input_shape=(
                    22,), kernel_regularizer=l_regtypes[regt1](regv1)))
            else:
                model.add(tf.keras.layers.Dense(
                    22, activation="relu", input_shape=(22,)))
            model.add(tf.keras.layers.Dropout(d1)) if d1 != 0 else None
            model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

            # compile model
            model.compile(optimizer=opt, loss=lloss, metrics=["accuracy"])

            print("Training")
            # train model
            # print training accuracy every 10 epochs

            class PrintAccuracy(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs={}):
                    if epoch % 10 == 0:
                        print("Training accuracy: ", logs.get("accuracy"))

            model.fit(training_features, training_labels,
                    epochs=epoch, callbacks=[PrintAccuracy()])

        # evaluate model
        em = model.evaluate(testing_features, testing_labels)
        acc = em[1]
        loss = em[0]
        em = model.evaluate(training_features, training_labels)
        test_acc = em[1]
        test_loss = em[0]

        # save model results to file
        print("Saving")
        print("Model: ", model)
        print("Optimizer: ", opt)
        print("Loss: ", lloss)
        print("Epochs: ", epoch)
        print("Dropout1: ", d1)
        # print("Dropout2: ", d2)
        print("Batch Normalization: ", norm)
        print("Regularization 1: ", reg1)
        print("Regularization Type 1: ", regt1)
        print("Regularization Value 1: ", regv1)
        # print("Regularization 2: ", reg2)
        # print("Regularization Type 2: ", regt2)
        # print("Regularization Value 2: ", regv2)
        print("Training Accuracy: ", test_acc)
        print("Testing Accuracy: ", acc)
        print("Training Loss: ", test_loss)
        print("Testing Loss: ", loss)
        f.write(str(count)+", "+str(model)+", "+str(opt)+", "+str(lloss)+", "+str(epoch)+", "+str(d1)+", "+str(norm)+", " +
                str(reg1)+", "+str(regt1)+", "+str(regv1)+", "+str(test_acc)+", "+str(acc)+", "+str(test_loss)+", "+str(loss)+"\n")
        f.flush()

        # save model
        if not loaded:
            model.save("models/mushroom_model_"+str(count))
            model.save_weights("weights/mushroom_model_weights_"+str(count))
            print('Saved mushroom_model_'+str(count))
        count = count + 1
        loaded = False
