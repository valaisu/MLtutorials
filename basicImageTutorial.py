# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print(train_images.shape)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']#names for labels instead of numbers

#demo ekasta kuvasta
'''plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()'''

train_images = train_images / 255.0 #scale rpg values to 0-1 range

test_images = test_images / 255.0

#demo 25 ekasta kuvasta mustavalkosena
'''plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)#jännä et tää puolierrori ei haittaa
    plt.xlabel(class_names[train_labels[i]])
plt.show()'''

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),#muuttaa 28x28 kuvan 784 pitkäks arrayks
    tf.keras.layers.Dense(2048, activation='relu'),#128 neuronia, koitetaan oisko se fiksumpi jos niitä ois enemmän
    #miettimisaika kasvo lineaarisesti, mut tulokset ei juuri muuttunu
    #512-2048 8 kertasti miettimisajan
    #10-kertanen aivokapasiteetti ja 2-kertanen treenausaika teki tästä hyvin marginaalisesti paremman
    tf.keras.layers.Dense(10)#palaute 10 pitkä "logits array" (hämmentävää)
])#ton muuttaminen 10->40 hajottaa, poitti varmaan et mahollisii outputteja on 10

model.compile(optimizer='adam',#avainsana compile
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


model.fit(train_images, train_labels, epochs=20)#joo 10 epochia


test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)#koitetaan kuvia joita ei oo käytetty treenamiseen

print('\nTest accuracy:', test_acc)


#seuraavaks käytetään treenattua modelia
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)

print(np.argmax(predictions[0]))


#katotaan 10 ekan kuvan arvaukset
def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

#katotaan yks kuva
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

#katotaan monta kuvaa
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()


#ööm
img = test_images[1]

img = (np.expand_dims(img,0)) #kuvat pitää aina laittaa jostain syystä listaan
predictions_single = probability_model.predict(img)

print(predictions_single)

plot_value_array(1, predictions_single[0], test_labels)#pitäisköhän tän tehä jtn
_ = plt.xticks(range(10), class_names, rotation=45)
