import os
import tensorflow as tf
import skimage
from skimage import transform
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt

def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory)
                      if f.endswith(".ppm")]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))

    return np.array(images), np.array(labels)

ROOT_PATH = "/home/dean/Github/Machine-Learning/BelgiumTS/"
train_data_directory = os.path.join(ROOT_PATH, "Data/Training")
test_data_directory = os.path.join(ROOT_PATH, "Data/Testing")

images, labels = load_data(train_data_directory)

print(images.ndim)
print(images.size)

print(labels.ndim)
print(labels.size)
print(len(set(labels)))


'''
plt.hist(labels, 62)
plt.show()

traffic_signs = [300, 2250, 3650, 4000]
for i in range(len(traffic_signs)):
    plt.subplot(1, 4, i+1)
    plt.axis('off')
    plt.imshow(images[traffic_signs[i]])
    plt.subplots_adjust(wspace=0.5)
    plt.show()
    print("shape: {0}, min: {1}, max: {2}".format(images[traffic_signs[i]].shape,
                                                  images[traffic_signs[i]].min(),
                                                  images[traffic_signs[i]].max()))
'''
images28 = np.array([transform.resize(image, (28, 28)) for image in images])
images28 = rgb2gray(images28)

x = tf.placeholder(dtype = tf.float32, shape = [None, 28, 28])
y = tf.placeholder(dtype = tf.int32, shape = [None])

images_flat = tf.contrib.layers.flatten(x)

logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits))

train_op = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss)

correct_pred = tf.argmax(logits, 1)

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

print("images_flat: ", images_flat)
print("logits: ", logits)
print("loss: ", loss)
print("predicted_labels: ", correct_pred)

tf.set_random_seed(1234)
sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(1000):
    print('EPOCH', i)
    _, accuracy_val = sess.run([train_op, accuracy], feed_dict = {x:images28, y: labels})
    if i % 10 == 0:
        print("Loss: ", loss)
    print("Done iteration: ", i)

import random
'''
sample_indexes = random.sample(range(len(images28)), 10)
sample_images = [images28[i] for i in sample_indexes]
sample_labels = [labels[i] for i in sample_indexes]

predicted = sess.run([correct_pred], feed_dict={x: sample_images})[0]
print(sample_labels)
print(predicted)

fig = plt.figure(figsize=(10, 10))
for i in range(len(sample_images)):
    truth = sample_labels[i]
    prediction = predicted[i]
    plt.subplot(5, 2,1+i)
    plt.axis('off')
    color='green' if truth == prediction else 'red'
    plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction),
             fontsize=12, color=color)
    plt.imshow(sample_images[i],  cmap="gray")

plt.show()
'''
test_images, test_labels = load_data(test_data_directory)

test_images28 = [transform.resize(image, (28, 28)) for image in test_images]

test_images28 = rgb2gray(np.array(test_images28))

predicted = sess.run([correct_pred], feed_dict={x: test_images28})[0]

match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])

accuracy = match_count / len(test_labels)

print("Accuracy: {:.3f}".format(accuracy))
