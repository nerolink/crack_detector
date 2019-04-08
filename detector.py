# In[1]
# python notebook for Make Your Own Neural Network
# code for a 3-layer neural network, and code for learning the MNIST dataset
# this version asks the network what the image should be, given a label
# (c) Tariq Rashid, 2016
# license is GPLv2

import numpy
# scipy.special for the sigmoid function expit(), and its inverse logit()
import scipy.special
import scipy.misc
import matplotlib.pyplot
import time
import os


# library for plotting arrays
# import matplotlib.pyplot

# ensure the plots are inside this notebook, not an external window
# %matplotlib inline

# neural network class definition
# 神经网络定义部分
class neuralNetwork:

    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc
        # 以正态分布建立初始权重，均值0.0，方差为节点数量的-0.5次方
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # learning rate
        self.lr = learningrate

        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)
        self.inverse_activation_function = lambda x: scipy.special.logit(x)

        pass

    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))

        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))

        pass

    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T

        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    # backquery the neural network
    # we'll use the same termnimology to each item,
    # eg target are the values at the right of the network, albeit used as input
    # eg hidden_output is the signal to the right of the middle nodes
    def backquery(self, targets_list):
        # transpose the targets list to a vertical array
        final_outputs = numpy.array(targets_list, ndmin=2).T

        # calculate the signal into the final output layer
        final_inputs = self.inverse_activation_function(final_outputs)

        # calculate the signal out of the hidden layer
        hidden_outputs = numpy.dot(self.who.T, final_inputs)
        # scale them back to 0.01 to .99
        hidden_outputs -= numpy.min(hidden_outputs)
        hidden_outputs /= numpy.max(hidden_outputs)
        hidden_outputs *= 0.98
        hidden_outputs += 0.01

        # calculate the signal into the hidden layer
        hidden_inputs = self.inverse_activation_function(hidden_outputs)

        # calculate the signal out of the input layer
        inputs = numpy.dot(self.wih.T, hidden_inputs)
        # scale them back to 0.01 to .99
        inputs -= numpy.min(inputs)
        inputs /= numpy.max(inputs)
        inputs *= 0.98
        inputs += 0.01

        return inputs


# 定义输入信息
# number of input, hidden and output nodes
input_nodes = 160000
width = 1600
height = 100

hidden_nodes = 1000
output_nodes = 8
input_path = 'input_data'
output_path = 'output_data'
data_floor = 1
data_ceiling = 27

# learning rate(学习率，此处可以修改)
learning_rate = 0.1

# create instance of neural network 建立神经网络
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# 以世代方式循环数据并提前处理
# epochs is the number of times the training data set is used for training
epochs = 10  # 世代数
# time=[1,2,3,4,5,6,7,8]
# time=range(1,27)

# 下面这一行起到循环的作用
for e in range(epochs):
    #    # go through all records in the training data set
    for record in range(data_floor, data_ceiling):
        # split the record by the ',' commas
        img_array = scipy.misc.imread('%s/%d.png' % (input_path, record), flatten=True)

        img_data = 255.0 - img_array.reshape(input_nodes)
        inputs = (numpy.asfarray(img_data[:]) / 255.0 * 0.99) + 0.01

        targets = numpy.zeros(output_nodes) + 0.01  # 生成一行八列的0矩阵，0为浮点数，然后在0的基础上加上0.01

        resultnumber = [0, 4, 1, 4, 5, 5, 2, 3, 1, 1, 1, 1, 2, 2, 1, 2, 2, 2, 7, 6, 5, 6, 5, 6, 6, 4]  # 0最低7最高

        targets[resultnumber[record - 1]] = 0.99
        n.train(inputs, targets)
        pass
    pass

# In[2]

# 向后输出部分
# run the network backwards, given a label, see what image it produces
# label to test
label = 6  # 这里好像只能一次一次的输出
# create the output signals for this label
targets = numpy.zeros(output_nodes) + 0.01
# all_values[0] is the target label for this record
targets[label] = 0.99
print(targets)

# get image data
image_data = n.backquery(targets)

# plot image data
matplotlib.pyplot.imshow(image_data.reshape(height, width), cmap='Greys', interpolation='None')

# 可以采用分节运行的方式向后输出每个指标对应的底板裂缝情况


# In[3]
# 为什么这个循环只能输出一次图像？循坏并没有起到多次输出的效果
label = 0

import matplotlib.pyplot as plt

image_path = '%s/%s' % (output_path, time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time())))
if not os.path.exists(image_path):
    os.mkdir(image_path)
while label < 7:
    targets = numpy.zeros(output_nodes) + 0.01
    targets[label] = 0.99
    print(label)
    image_data = n.backquery(targets)
    image_data = image_data.reshape(height, width)
    plt.imsave('%s/%s.png' % (image_path, label), image_data, cmap='Greys')
    label = label + 1

# 如果加入支座信息是否需要变成ELman神经网络？
