import matplotlib.pyplot as plt
import tensorflow as tf
import urllib
import numpy as np
import zipfile
import os
from scipy.io import wavfile
from skimage.data import coffee
from skimage.transform import resize as imresize
import pickle

labels = np.array([[1,0,0,0,0],[0,1,0,0,0],[0,1,0,0,0],[0,1,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0],[0,0,1,0,0],[0,1,0,0,0],[1,0,0,0,0],[0,0,0,1,0],[0,0,0,1,0],[0,1,0,0,0],[0,0,0,0,1],[0,0,0,1,0],[0,0,0,0,1],[0,0,0,0,1],[0,0,1,0,0],[1,0,0,0,0],[0,1,0,0,0],[0,0,0,1,0],[0,0,0,1,0],[0,0,1,0,0],[0,0,0,0,1],[0,0,0,0,1],[0,0,1,0,0],[0,0,1,0,0],[1,0,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1],[0,1,0,0,0],[0,0,0,1,0],[0,0,0,0,1],[1,0,0,0,0],[0,1,0,0,0],[1,0,0,0,0]])


n_input = 25088
# The number of classes which the ConvNet has to classify into .
n_classes = 5
# The number of neurons in the each Hidden Layer .
n_hidden1 = 500
n_hidden2 = 500

def get_vgg_model():
    # download('https://s3.amazonaws.com/cadl/models/vgg16.tfmodel')
    with open("vgg16.tfmodel", mode='rb') as f:
        graph_def = tf.GraphDef()
        try:
            graph_def.ParseFromString(f.read())
        except:
            print('try adding PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python ' +
                  'to environment.  e.g.:\n' +
                  'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python ipython\n' +
                  'See here for info: ' +
                  'https://github.com/tensorflow/tensorflow/issues/582')

    # download('https://s3.amazonaws.com/cadl/models/synset.txt')
    # with open('synset.txt') as f:
    #     labels = [(idx, l.strip()) for idx, l in enumerate(f.readlines())]

    return {
        'graph_def': graph_def
        # 'labels': labels
        # 'preprocess': preprocess,
        # 'deprocess': deprocess
    }

def preprocess(img, crop=True, resize=True, dsize=(224, 224)):
    if img.dtype == np.uint8:
        img = img / 255.0

    if crop:
        short_edge = min(img.shape[:2])
        yy = int((img.shape[0] - short_edge) / 2)
        xx = int((img.shape[1] - short_edge) / 2)
        crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    else:
        crop_img = img

    if resize:
        norm_img = imresize(crop_img, dsize, preserve_range=True)
    else:
        norm_img = crop_img

    return (norm_img).astype(np.float32)
def deprocess(img):
    return np.clip(img * 255, 0, 255).astype(np.uint8)
    # return ((img / np.max(np.abs(img))) * 127.5 +
    #         127.5).astype(np.uint8)

net = get_vgg_model()

# labels = net['labels']

g1 = tf.Graph()

with tf.Session(graph=g1) as sess, g1.device('/cpu:0'):
    tf.import_graph_def(net['graph_def'], name='vgg')
    names = [op.name for op in g1.get_operations()]
# print names

g2 = tf.Graph()
with g2.as_default():

    # Tensorflow Graph input .
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])

    with tf.name_scope('layer1'):
        W_1 = tf.get_variable(
                    name="W1",
                    shape=[n_input, n_hidden1],
                    dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())

        b_1 = tf.get_variable(
            name='b1',
            shape=[n_hidden1],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0))

        h_1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(x, W_1),b_1))

    with tf.name_scope('layer2'):
        W_2 = tf.get_variable(
                    name="W2",
                    shape=[n_hidden1,n_hidden2],
                    dtype=tf.float32,
                    initializer=tf.contrib.layers.xavier_initializer())

        b_2 = tf.get_variable(
            name='b2',
            shape=[n_hidden2],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0))

        h_2 = tf.nn.relu(tf.nn.bias_add(
            tf.matmul(h_1, W_2),b_2))

    with tf.name_scope('output'):
       W_3 = tf.get_variable(
                   name="W3",
                   shape=[n_hidden2,n_classes],
                   dtype=tf.float32,
                   initializer=tf.contrib.layers.xavier_initializer())

       b_3 = tf.get_variable(
           name='b3',
           shape=[n_classes],
           dtype=tf.float32,
           initializer=tf.constant_initializer(0.0))

       h_3 = tf.nn.bias_add(tf.matmul(h_2, W_3),b_3)

    # Y_pred = tf.nn.softmax(h_3)
    Y_pred = h_3

    Cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Y_pred, y))
    optimizer = tf.train.AdamOptimizer(0.01).minimize(Cost)

    #Monitor accuracy
    predicted_y = tf.argmax(Y_pred, 1)
    actual_y = tf.argmax(y, 1)

    correct_prediction = tf.equal(predicted_y, actual_y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    names = [op.name for op in g2.get_operations()]
    # print names



for i in range(401,439):
    x1 = g1.get_tensor_by_name('vgg/images' + ':0')
    img=[]
    og = plt.imread("images/"+str(i)+".png")
    og = preprocess(og)
    img.append(og)
    img_4d = np.array(img)
    # img_4d = img_4d.reshape((1,224,244,3))
    # img_4d = img[np.newaxis]

    print img_4d.shape , "Image Shape"

    with tf.Session(graph=g1) as sess, g1.device('/gpu:0'):


            content_layer = 'vgg/pool5:0'
            content_features= g1.get_tensor_by_name(content_layer).eval(
                    session=sess,
                    feed_dict={x1: img_4d,
                        'vgg/dropout_1/random_uniform:0': [[1.0]],
                        'vgg/dropout/random_uniform:0': [[1.0]]
                    })

            # train_new.append(content_features)
            print content_features.shape

    new_input = content_features
    new_input = new_input.reshape((new_input.shape[0],7*7*512))
    print new_input.shape , "Feature Map Shape"

    label = labels[i-401].reshape(1,5)
    print label.shape

    with tf.Session(graph=g2) as sess, g2.device('/gpu:0'):
        sess.run(tf.initialize_all_variables())
        n_epochs=3
        # training
        for epoch in range(n_epochs):

            sess.run(optimizer, feed_dict={x: new_input, y:label})

        # print str(epoch) + "-------------------------------------"
        print(sess.run(accuracy, feed_dict={x: new_input,y: label}))
