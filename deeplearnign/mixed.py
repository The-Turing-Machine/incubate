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

train_new = []
def download(path):
    """Use urllib to download a file.
    Parameters
    ----------
    path : str
        Url to download
    Returns
    -------
    path : str
        Location of downloaded file.
    """
    import os
    from six.moves import urllib

    fname = path.split('/')[-1]
    if os.path.exists(fname):
        return fname

    print('Downloading ' + path)

    def progress(count, block_size, total_size):
        if count % 20 == 0:
            print('Downloaded %02.02f/%02.02f MB' % (count * block_size / 1024.0 / 1024.0,total_size / 1024.0 / 1024.0))

    filepath, _ = urllib.request.urlretrieve(
        path, filename=fname, reporthook=progress)
    return filepath


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

    download('https://s3.amazonaws.com/cadl/models/synset.txt')
    with open('synset.txt') as f:
        labels = [(idx, l.strip()) for idx, l in enumerate(f.readlines())]

    return {
        'graph_def': graph_def,
        'labels': labels
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

labels = net['labels']

g = tf.Graph()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# session = tf.Session(config=config, ...)

with tf.Session(graph=g,config=config) as sess, g.device('/cpu:0'):
    tf.import_graph_def(net['graph_def'], name='vgg')
    names = [op.name for op in g.get_operations()]

# for i in names:
#     print i


#

# for i in range(401,501):
#     og = plt.imread("images/"+str(i)+".png")
#     og = preprocess(og)
#     img.append(og)
#
# img = np.array(img)
# print img.shape
# # plt.imshow(og)
# # plt.show()
#
#
# # print img.shape
# # plt.imshow(deprocess(img))
# # plt.show()
# img_4d = img
# img_4d = img.reshape(5,224,224,3)
# img_4d = img[np.newaxis]
# print img_4d.shape

x = g.get_tensor_by_name(names[0] + ':0')
softmax = g.get_tensor_by_name(names[-2] + ':0')
# print softmax
# To get the feature map
for i in range(401,405):
    img=[]
    og = plt.imread("images/"+str(i)+".png")
    og = preprocess(og)
    img.append(og)
    img_4d = np.array(img)
    # img_4d = img_4d.reshape((1,224,244,3))
    # img_4d = img[np.newaxis]

    print img_4d.shape

    with tf.Session(graph=g) as sess, g.device('/cpu:0'):


            content_layer = 'vgg/pool5:0'
            content_features= g.get_tensor_by_name(content_layer).eval(
                    session=sess,
                    feed_dict={x: img_4d,
                        'vgg/dropout_1/random_uniform:0': [[1.0]],
                        'vgg/dropout/random_uniform:0': [[1.0]]
                    })

            # train_new.append(content_features)
            print content_features.shape


    new_input = content_features
    print new_input.shape

    labels = np.array([[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0]])
    print labels.shape

    n_input = 25088
    # The number of classes which the ConvNet has to classify into .
    n_classes = 5
    # The number of neurons in the each Hidden Layer .
    n_hidden1 = 4096
    n_hidden2 = 4096

    g = tf.get_default_graph()

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

    Cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Y_pred, labels))
    optimizer = tf.train.AdamOptimizer(0.001).minimize(Cost)

    #Monitor accuracy
    predicted_y = tf.argmax(Y_pred, 1)
    actual_y = tf.argmax(labels, 1)

    correct_prediction = tf.equal(predicted_y, actual_y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    o = [op.name for op in g.get_operations()]
    for i in o:
        print i

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    saver = tf.train.Saver()

    sess = tf.Session(config=config)
    sess.run(tf.initialize_all_variables())

    n_epochs=3
    # training
    for epoch in range(n_epochs):
        sess.run(optimizer, feed_dict={x: new_input, y:labels})


        print str(epoch) + "-------------------------------------"
        print(sess.run(accuracy, feed_dict={x: new_input,y: labels}))
