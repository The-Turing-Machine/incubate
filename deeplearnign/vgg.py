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

with tf.Session(graph=g,config=config) as sess, g.device('/gpu'):
    tf.import_graph_def(net['graph_def'], name='vgg')
    names = [op.name for op in g.get_operations()]

for i in names:
    print i


#
img=[]
for i in range(401,402):
    og = plt.imread("images/"+str(i)+".png")
    og = preprocess(og)
    img.append(og)

img = np.array(img)
print img.shape
# plt.imshow(og)
# plt.show()


# print img.shape
# plt.imshow(deprocess(img))
# plt.show()
img_4d = img
# img_4d = img.reshape(5,224,224,3)
# img_4d = img[np.newaxis]
print img_4d.shape

x = g.get_tensor_by_name(names[0] + ':0')
softmax = g.get_tensor_by_name(names[-2] + ':0')
# print softmax
# To get the feature map
with tf.Session(graph=g) as sess, g.device('/gpu'):

    content_layer = 'vgg/pool5:0'
    content_features= g.get_tensor_by_name(content_layer).eval(
            session=sess,
            feed_dict={x: img_4d,
                'vgg/dropout_1/random_uniform:0': [[1.0]],
                'vgg/dropout/random_uniform:0': [[1.0]]
            })
    # soft = g.get_tensor_by_name('vgg/prob:0').eval(
    #         session=sess,
    #         feed_dict={x: img_4d
    #         })
    # print content_features[0,:,:,1].shape
    # print soft.shape
    # maxs = np.argsort(soft[0],0)[:5]
    # print maxs
    train_new.append(content_features)

# print(content_features.shape)
print(np.array(train_new[0]).shape)


# plt.imshow(content_features[0,:,:,1],cmap="gray")
# plt.show()

#Visualize last layer

# for i in range(1,100):
#     plt.subplot(10,10,i)
#     og = content_features[0,:,:,i]
#     plt.imshow(og,cmap="gray")
# plt.show()

file_Name = "input_new"
# open the file for writing
fileObject = open(file_Name,'wb')

# this writes the object a to the
# file named 'testfile'
pickle.dump(np.array(train_new[0]),fileObject)

# here we close the fileObject
fileObject.close()
