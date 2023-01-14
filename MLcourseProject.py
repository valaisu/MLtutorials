'''
AT,V,AP,RH,PE
Air Temperature (AT),
Ambient Pressure (AP),
Relative Humidity (RH) and
Exhaust Vacuum (V) to predict the
net hourly electrical energy output (EP)  of the plant
'''

# The course projet of a basic machine learning course
# analyzes power output of a powerplant based on five
# other variables listed above


import matplotlib.pyplot as plt
import math
from numpy import loadtxt
import tensorflow as tf
from tensorflow.keras import layers

dataset = loadtxt('Folds5x2_pp.csv', delimiter=',')
features = dataset[:,0:4] #other data
labels = dataset[:,4] #labels
a = 0
b = 450
c = 450
for i in range(len(labels)):
    a += labels[i]
    if labels[i] < b:
        b = labels[i]
    if labels[i] > c:
        c = labels[i]

print(a/len(labels))
print("pienin: " + str(b) + " suurin: " + str(c))

plant_model = tf.keras.Sequential([
  layers.Dense(4),#too many neurons is bad, sweetspot somewhere between 16 and 125
  #layers.Dense(4), #doesn't really make any difference, for the better anyway
  layers.Dense(1)#output has 1 parameter
])

plant_model.compile(loss = tf.losses.MeanAbsoluteError(),#Bit better than mean^2 (4 vs sqrt(30))
                    optimizer = tf.optimizers.Adam(),
                    metrics=['accuracy'])#saves data for later

history = plant_model.fit(features, labels, epochs=40, validation_split=0.2)#20% of data goes to validation



#Results vizualization
r = range(40)

loss = history.history['loss']
val_loss = history.history['val_loss']
for i in range(len(loss)):
    loss[i] = math.sqrt(loss[i])
    val_loss[i] = math.sqrt(val_loss[i])

axes = plt.gca()
axes.set_ylim([0, 20])
plt.plot(r, loss, label='Training Loss')
plt.plot(r, val_loss, label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Mean Absolute Error, 4 Neurons, Optimizer Adam")
plt.legend()
plt.show()
