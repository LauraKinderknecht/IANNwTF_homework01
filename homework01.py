import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

###
# 2.1 Data
###

#load the dataset
digits = load_digits()

#extract into (input, target) tuples
#save "pixels"-array in "data"
data = digits.data
#save targets in "target"
target = digits.target
#zip the data and targets tuples into a list
data_tuples = list(zip(data, target))
#prints the type of the first elements in data and the first element itself
#print(type(data[0]), data[0][:])
#print()

'''
#prints the shape of data
print("Data shape: ", data.shape)
#prints the shape of our targets
print("Targets shape: ", target.shape)
print()
'''

#plot first 10 data images
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
#go through the first tuples in data_tuples
for i, (image, t) in enumerate(data_tuples[:10]):
    #determine the row and column in plot
    ax = axes[i // 5, i % 5]
    #plot image
    ax.imshow(image.reshape(8, 8), cmap='gray')
    #show no axis
    ax.axis('off')


#images have the shape (64)
#print("Image shapes: ",np.shape(data))

#change type into float32 and reshape with maximum pixel value
data = data.astype(np.float32)
data = data / 16.0
#print("Data type: ", type(data[0][0]))

#onehot encoding target
targets = []
#interate through all target values
for j in target:
    #create a one-dimensional array of ten zeros
    one_hot_target = np.zeros(10) 
    #at idx j of one_hot_targets, replace 0.0 with 1.0
    one_hot_target[j] = 1.0
    #append to targets array
    targets.append(one_hot_target)
target = targets
#print("First target with one-hot encoding: ", target[0])

#zip tuples with data and one-hot-encoded target into a list
data_tuples = list(zip(data, target))

#generate minibatches
def batch_generator(data_tuples, minibatch_size):
    #shuffle the tuples in the data_tuples list
    random.shuffle(data_tuples)
    #determine the number of tuples
    num_samples = len(data_tuples)
    #determine the number of minibatches depending on the desired minibatch size
    num_minibatches = num_samples // minibatch_size 

    # for the number of minibatches, do:
    for i in range(num_minibatches):
        #determine start index
        start_idx = i * minibatch_size
        #determine end index
        end_idx = (i + 1) * minibatch_size
        #using the start and end index determine the tuples from data_tuples that will be in the minibatch
        minibatch_data = [data for data, target in data_tuples[start_idx:end_idx]]
        minibatch_target = [target for data, target in data_tuples[start_idx:end_idx]]

    return np.array(minibatch_data), np.array(minibatch_target)

'''
#create a minibatch - example

data = batch_generator(data_tuples, 5 )
print("data: ")
print()
#prints the inputs
print(data[0])
#prints the targets
print(data[1])  #gibt die targets aus
'''


###
# 2.2 Sigmoid Activation Function
###
class SigmoidActivation():
    def __init__(self):
        pass

    #actual sigmoid function
    def __call__(self,x):
        #save the input as self.input
        self.input = x
        #apply sigmoid function and return the solution
        return 1 / (1 + np.exp(-x))
    
    #using the sigmoid derivative function 
    #calculate the backwards step
    def backward(self, grad_output):
        sigmoid_grad = self.__call__(self.input) * (1 - self.__call__(self.input))
        return grad_output * sigmoid_grad

'''
#example for sigmoid activation with a minibatch of size 5
sig_activate = SigmoidActivation()
print()
print("Sigmoid: ")
print(sig_activate(batch_generator(data_tuples, 5)[0]))
'''

###
# 2.3 Softmax Activation Function
###
class SoftmaxActivation():
    def __init__(self):
        pass

    #actual softmax activation function
    def __call__(self, x):

        softmax_output = []
        #get the size of the input vector - len(first elem in x) = 10, 
        #because there are 10 different digits
        size_input_vector = len(x[0])

        #iterate through every target vector
        for row in x:
            
            #calculate e^{z_i} for every element z_i in the current target vector
            row_exponents = np.exp(row)
            #calculate the sum of e^_{z_j} for j = 1,…,K with K=10 - so for all e^{z_i} elements in the target vector
            row_sum = np.sum(row_exponents, axis=-1, keepdims=True)
            
            #iterate through all elements in the current target vector
            for idx, elem in enumerate(row_exponents):
                #divide each e^{elem} be the sum of the vector
                row_exponents[idx] = row_exponents[idx]/row_sum
            
            #append converted vector to the softmax output array
            softmax_output.append(row_exponents)
        
        # Save the output for later use in backpropagation
        self.softmax_output = softmax_output  
        return softmax_output


#example for softmax activation
'''
soft_activate = SoftmaxActivation()

print()
print("Softmax:")
print(soft_activate(batch_generator(data_tuples, 5)[1]))
'''

###
# 2.4 MLP Layers and Weights
###

class MLP_layer():
    # input_size = number of units/perceptrons in the preceding layer
    # num_perceptrons = number of units/perceptrons in the given layer
    def __init__(self, input_size, num_perceptrons, activation = SigmoidActivation(), loc = 0.0, scale = 0.2):
    #def __init__(self, input_size, num_perceptrons, loc = 0.0, scale = 0.2):
    
        #initialize weights as small, random, normally distributed values
        self.weights = np.random.normal(loc = loc, scale = scale, size = (input_size, num_perceptrons))
        #initialize the bias values set to zero
        self.bias = np.zeros((1, num_perceptrons))

        if activation == SigmoidActivation():
            self.activation = SigmoidActivation()
        elif activation == SoftmaxActivation():
            self.activation = SoftmaxActivation()

    def forward(self, input):
        #apply matrix multiplication and multiply the input and the weights
        #then add the bias
        #print("weights: ", self.weights)
        #print("input: ", input)
        output = np.matmul(input, self.weights) + self.bias

        # !!!
        # here is where the problem is
        # there seems to be an issue with the self.activation - but we couldn't resolve it by now!
        # !!!

        output = self.activation(output)
        return output
    
    #for the weights do matrixmultiplication with the transposed input and the grad_output
    def backward_weights(self, grad_output, input):
        grad_weights = np.matmul(input.T, grad_output)
        return grad_weights

    #for the backward_input do matrix multiplication with the transposed weights and the grad_output
    def backward_input(self, grad_output):
        grad_input = np.matmul(grad_output, self.weights.T)
        return grad_input

    #for the backwaards step assign the grad_activation, grad_weights and grad_input and return grad_weights and grad_input
    def backward(self, grad_output, input):
        grad_activation = self.activation.backward(grad_output)
        grad_weights = self.backward_weights(grad_activation, input)
        grad_input = self.backward_input(grad_activation)
        return grad_input, grad_weights

###
# 2.5 Putting together the MLP
###

class MLP():
    def __init__(self, layer_sizes):
        self.layers = []
        
        #go through all layers

        for i in range(len(layer_sizes)-1):
            hidden_layer = MLP_layer(input_size=layer_sizes[i], num_perceptrons=layer_sizes[i+1], activation=SigmoidActivation())
            self.layers.append(hidden_layer)
        
        self.layers[-1].activation = SoftmaxActivation()
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, x, target, loss_obj):
        # Initialize a list of dictionaries to store activations, pre-activations, and gradients for each layer
        tape = [{} for _ in range(len(self.layers))]

        # Forward pass
        for i, layer in enumerate(self.layers):
            x = layer.forward(x)
            tape[i]['input'] = x
            tape[i]['pre_activation'] = np.matmul(tape[i]['input'], layer.weights) + layer.bias

        # Compute CCE loss and its gradient
        loss = loss_obj.loss_calculation(x, target)
        grad_loss = loss_obj.backward(x, target)

        # Backward pass
        for i in reversed(range(len(self.layers))):
            grad_input, grad_weights = self.layers[i].backward(grad_loss, tape[i]['input'])
            grad_loss = grad_input

            tape[i]['grad_weights'] = grad_weights

        return loss, tape
    

###
# 2.6 CCE loss function
###

class CCE_Loss():
    def __init__(self):
        pass

    #function input are the made predictions and the targets
    def loss_calculation(pred, target):
        #using the provided formula
        #calculate the cce loss
        prob = target * np.log(pred)
        cce_result = -1 * np.sum(prob, axis=1)
        return cce_result
    
    def backward(self, pred, target):
        grad_loss = np.array(pred) - np.array(target)
        return grad_loss

#training function
def train(model, data_tuples, minibatch_size, epochs, learning_rate):
    loss_obj = CCE_Loss()
    losses = []

    #iterate through the epoches
    for epoch in range(epochs):

        #start with a total loss of 0
        total_loss = 0

        #iterate through all minibatches
        for _ in range(len(data_tuples) // minibatch_size):
            minibatch_data, minibatch_target = batch_generator(data_tuples, minibatch_size)
            #calculate the predictions with a forward run through the MLP
            predictions = model.forward(minibatch_data)
            #compute loss and tape with backpropagation
            loss, tape = model.backward(predictions, minibatch_target, loss_obj)

            #add the mean loss to the total loss
            total_loss += np.mean(loss)

            # Update weights using gradient descent
            for i, layer in enumerate(model.layers):
                layer.weights -= learning_rate * np.mean(tape[i]['grad_weights'], axis=0)
                layer.bias -= learning_rate * np.mean(grad_loss, axis=0)

        #calculate the average loss over the minibatches
        average_loss = total_loss / (len(data_tuples) // minibatch_size)
        losses.append(average_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss}")

    # Plotting loss vs. epoch
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss vs. Epoch')
    plt.show()

#training a MLP
train(model = MLP(layer_sizes=[64, 32, 32, 10]), data_tuples=data_tuples, minibatch_size=5, epochs=100, learning_rate=0.1)