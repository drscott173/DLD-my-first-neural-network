import numpy as np

class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        # Sigmoid activation
        self.activation_function = lambda x : 1/(1+np.exp(-x))

    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            
            # Implement the forward pass function below
            final_outputs, hidden_outputs = self.forward_pass_train(X)  

            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)

        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch

        '''
        # Hidden layer, sigmoid activation
        hidden_inputs = np.dot(X, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)

        # Output layer, regression
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
        final_outputs = final_inputs

        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        # Output error, no activation
        error = (y-final_outputs)
        output_error_term = error
        
        # Hidden layer error, sigmoid activation
        hidden_error = np.dot(self.weights_hidden_to_output, output_error_term)
        hidden_error_term = hidden_error * hidden_outputs * (1-hidden_outputs)
        
        # Weight step (hidden to output)
        delta_weights_h_o += output_error_term * hidden_outputs[:,None]

        # Weight step (input to hidden)
        delta_weights_i_h += hidden_error_term * X[:, None]
        
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records

    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        final_outputs, _ = self.forward_pass_train(features)
        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 10000
learning_rate = 0.96
hidden_nodes = 12
output_nodes = 1
print("Hi there :-)")
 
# Experiments

# lr=0.001, it=10k, nodes=64, train=0.64, val=1.2
# lr=0.005, it=4k, nodes=48, train=0.58 val=0.97
# lr=0.01, it=4k, nodes=48, train=0.47 val=0.79
# lr=0.02, it=4k, nodes=48, train=0.45 val=0.75
# lr=0.02, it=4k, nodes=32, train=0.33 val=0.52
# lr=0.04, it=4k, nodes=32, train=0.29 val=0.46
# lr=0.08, it=4k, nodes=32, train=0.27 val=0.43
# lr=0.08, it=4k, nodes=16, train=0.27 val=0.45
# lr=0.08, it=4k, nodes=12, train=0.26 val=0.44 
# lr=0.08, it=4k, nodes=8, train=0.26 val=0.44 
# lr=0.08, it=4k, nodes=7, train=0.26 val=0.42 
# lr=0.08, it=4k, nodes=6, train=0.26 val=0.43
# lr=0.16, it=4k, nodes=7, train=0.21 val=0.36
# lr=0.32, it=4k, nodes=7, train=0.11 val=0.24 
# lr=0.64, it=10k, nodes=7, train=0.07 val=0.18 
# lr=0.96, it=10k, nodes=7, train=0.06 val=0.15 
# lr=0.99, it=4k, nodes=7, train=0.07 val=0.17 

# lr=0.32, it=10k, nodes=7, train=0.07 val=0.16
# lr=0.64, it=4k, nodes=7, train=0.08 val=0.18
# lr=0.96, it=4k, nodes=7, train=0.06 val=0.15 
# lr=0.96, it=10k, nodes=7, train=0.06 val=0.15 
# lr=0.08, it=10k, nodes=7, train=0.16 val=0.32
# lr=0.16, it=10k, nodes=7, train=0.07 val=0.14 ***
# lr=0.16, it=10k, nodes=7, train=0.09 val=0.17  with offday
# lr=0.16, it=10k, nodes=12, train=0.07 val=0.15  with offday
# lr=0.16, it=10k, nodes=7, train=0.08 val=0.17  with offday
# lr=0.16, it=10k, nodes=12, train=0.08 val=0.22  with offday

# lr=0.16, it=10k, nodes=9, train=0.07 val=0.15  with offday ****
# lr=0.16, it=10k, nodes=9, train=0.08 val=0.19  with offday ****

# lr=0.16, it=10k, nodes=7, train=0.08 val=0.19



