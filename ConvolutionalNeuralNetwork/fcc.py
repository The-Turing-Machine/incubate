import tensorflow as tf
import pickle
import numpy as np
file_Name = "../deeplearnign/input_new"
fileObject = open(file_Name,'r')
# load the object from the file into var b
new_input = pickle.load(fileObject)
new_input = new_input.reshape((new_input.shape[0],7*7*512))
print new_input.shape

labels = np.array([[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0]])
print labels.shape
#
# Initialising the network parameters .
# The dimensions of the feature map for each image coming out of VGG Net flattened to a coloumn vector .
n_input = 7*7*512
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

# o = [op.name for op in g.get_operations()]
# for i in o:
#     print i

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

#
# #testing
#
#
# acc,prob = sess.run([accuracy,predicted_y],feed_dict={X: mnist.test.images,Y: mnist.test.labels})
# # save_path = saver.save(sess, "/home/ayush/Documents/neuralnetwork/model.ckpt")
# print prob.shape
# print acc

# # o = [op.name for op in g.get_operations()]
# # for i in o:
# #     print i
# W = g.get_tensor_by_name('W:0')
# W_arr = np.array(W.eval(session=sess))
# print(W_arr.shape)
# print W_arr[:,0]
# plt.imshow(W_arr[:,0].reshape(28,28),cmap='gray')
# plt.show()
# # fig, ax = plt.subplots(1, 10, figsize=(20, 3))
# # for col_i in range(10):
# #     ax[col_i].imshow(W_arr[:, col_i].reshape((28, 28)), cmap='coolwarm')
# # plt.show()
