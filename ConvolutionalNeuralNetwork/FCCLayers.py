import tensorflow as tf
import pickle
import numpy as np

file_Name = "../deeplearnign/input_new"
fileObject = open(file_Name,'r')
# load the object from the file into var b
new_input = pickle.load(fileObject)
print new_input.shape
new_input = new_input.reshape((new_input.shape[0],7*7*512))
print new_input.shape

labels = np.array([[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0],[1,0,0,0,0]])
print labels.shape

## Initialising the network parameters .
# The dimensions of the feature map for each image coming out of VGG Net flattened to a coloumn vector .
n_inputs = 7*7*512
# The number of classes which the ConvNet has to classify into .
n_classes = 5
# The number of neurons in the each Hidden Layer .
n_hidden1 = 4096
n_hidden2 = 4096

# Tensorflow Graph input .
x = tf.placeholder("float", [None, n_inputs])
y = tf.placeholder("float", [None, n_classes])

g = tf.get_default_graph()

## Initialising the Weights .
Weights = {
	'W1' : tf.get_variable("W1", shape=[n_inputs, n_hidden1], initializer=tf.contrib.layers.xavier_initializer()),
	'W2' : tf.get_variable("W2", shape=[n_hidden1, n_hidden2], initializer=tf.contrib.layers.xavier_initializer()),
	'W3' : tf.get_variable("W3", shape=[n_hidden1, n_classes], initializer=tf.contrib.layers.xavier_initializer())
}

## Initialising the Bias .
Biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden1])),
    'b2': tf.Variable(tf.random_normal([n_hidden2])),
    'b3': tf.Variable(tf.random_normal([n_classes]))
}

## Creating the Model .

def Create_Model(x, Weights, Biases) :

	## First , we perform matrix multiplication of the input images with the first Weight matrix .
	Layer_1 = tf.add(tf.matmul(x, Weights['W1']), Biases['b1'])
	## We then apply the 'Relu' activation function .
	Layer_1 = tf.nn.relu(Layer_1)

	## We now apply matrix multiplication of the ouput of the first hidden layer with the second Weight matrix .
	Layer_2 = tf.add(tf.matmul(Layer_1, Weights['W2']), Biases['b2'])
	## We then apply the 'Relu' activation function .
	Layer_2 = tf.nn.relu(Layer_2)

	## We now multiply these outputs with the last Weight matrix accordingly to our number of classes to get their scores .
	Output = Layer_2 = tf.add(tf.matmul(Layer_2, Weights['W3']), Biases['b3'])

	return Output

Predicted_Scores = Create_Model(x, Weights, Biases)

## Using the Softmax Loss Function as our Cost Function .
Cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(Predicted_Scores, y))

## Optimising our loss function with Adam using a learning rate of 0.001 .
Optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(Cost)

## We will now test our model .
Predicted_y = tf.argmax(Predicted_Scores, 1)
Actual_y = tf.argmax(labels, 1)

Correct_Prediction = tf.equal(Predicted_y, Actual_y)
Accuracy = tf.reduce_mean(tf.cast(Correct_Prediction, "float"))


saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.initialize_all_variables())

n_epochs=3
# training
for epoch in range(n_epochs):
    sess.run(Optimizer, feed_dict={x: new_input, y:labels})


    print str(epoch) + "-------------------------------------"
    print(sess.run(Accuracy, feed_dict={x: new_input,y: labels}))