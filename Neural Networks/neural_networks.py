import numpy as np

class NeuralNetwork(object):
    def __init__(self, num_layers=2, num_classes=3, hidden_size=10, hidden_activation_fn="relu"):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.hidden_activation_fn = hidden_activation_fn
        self.num_classes = num_classes

    def initialize_weights(self, input_dim, std_dev=1e-2):
        """
        Initialize the weights of the model. The weights are initialized
        to small random values. Weights are stored in the variable dictionary
        named self.params.

        W: weight vector; has shape (D, C)
        b: bias vector; has shape (C,1)

        Inputs:
        - input_dim: (int) The dimension D of the input data.
        - std_dev: (float) Controls the standard deviation of the random values.
        """

        self.params = {}
        hidden_size = self.hidden_size
        num_classes = self.num_classes
        num_layers = self.num_layers
        #############################################################################
        # TODO: Initialize the weight and bias.                                     #
        #############################################################################
        self.params['W1'] = std_dev * np.random.randn(input_dim, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std_dev * np.random.randn(hidden_size, num_classes)
        self.params['b2'] = np.zeros(num_classes)            
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################


    def fully_connected_forward(self, X, W, b):
        """
        Computes the forward pass of a fully connected layer.

        A fully connected / affine / linear / dense layer applies a linear transformation
        of the incoming data: Wx + b.

        Inputs:
        - X: A numpy array of shape (N, D)
        - W: A numpy array of weights, of shape (D, M)
        - b: A numpy array of biases, of shape (M,)

        Returns a tuple of:
        - out: output of shape (N, M)
        - cache: (X, W, b)
        """

        #############################################################################
        # TODO: Implement the forward pass of a fully connected layer and store     #
        # the variables needed for the backward pass (gradient computation)         #
        # as a tuple inside cache.                                                  #
        #############################################################################
        out = X@W + b
        cache = (X, W, b)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

        return out, cache

    def fully_connected_backward(self, dUpper, cache):
        """
        Computes the backward pass for a fully connected layer layer.

        Inputs:
        - dUpper: Gradient of shape (N, M), coming from the upper layer.
        - cache: Tuple of:
            - X: A numpy array of shape (N, D)
            - W: A numpy array of weights, of shape (D, M)
            - b: A numpy array of biases, of shape (M,)

        Returns a tuple of:
        - dX: Gradient with respect to X, of shape (N, D)
        - dW: Gradient with respect to W, of shape (D, M)
        - db: Gradient with respect to b, of shape (M,)
        """
        X, W, b = cache
        dX, dW, db = None, None, None
        #############################################################################
        # TODO: Implement the affine backward pass.                                 #
        #############################################################################
        dX = dUpper@W.T
        dW = X.T@dUpper
        db = np.sum(dUpper,axis = 0)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return dX, dW, db


    def relu_forward(self, x):
        """
        Computes the forward pass of a rectified linear unit (ReLU).

        Input:
        - x: A numpy array / matrix of any shape

        Returns a tuple of:
        - out: A numpy array / matrix of the same shape as x
        - cache: x
        """
        out = None
        #############################################################################
        # TODO: Implement the ReLU forward pass.                                    #
        #############################################################################
        out = np.maximum(0,x)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        cache = x
        return out, cache


    def relu_backward(self,dout, cache):
        """
        Computes the backward pass for a layer of rectified linear units (ReLUs).

        Input:
        - dout: Upstream derivatives, of any shape
        - cache: Input x, of same shape as dout

        Returns:
        - dx: Gradient with respect to x
        """
        dx, x = None, cache.copy()
        #############################################################################
        # TODO: Implement the ReLU backward pass.                                   #
        #############################################################################
        dx = (x>0) * dout
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return dx

    def softmax(self,x):

        """
        Compute the softmax function for each row of the input x.

        Inputs:
        - x: A numpy array of shape (N, C) containing scores for each class; there are N
          examples each of dimension C.

        Returns:
        probs: A numpy array of shape (N, C) containing probabilities for each class.
        """

        #############################################################################
        # TODO: Implement the softmax function.                                     #
        #############################################################################
        e = np.exp(x - np.max(x, axis=1)[:,np.newaxis])
        if e.ndim == 1:
            probs = e / np.sum(e, axis=1)
        else:
            probs = e / np.array([np.sum(e, axis=1)]).T
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        
        return probs
        
    def softmax_cross_entropy_loss(self, scores, labels):
        """
            Jointly computes the softmax and cross entropy loss. This function should return
            the loss and its gradient with respect to the scores.

            Inputs:
            - scores: A numpy array of shape (N, C) containing scores for each class; there are N
              examples each of dimension C.
            - labels: A numpy array of shape (N,) containing the indices of the correct class for
              each example.

            Returns:
            loss: A scalar value corresponding to the softmax cross entropy loss
            dloss: A numpy array of shape (N, C) containing the gradients of the loss with respect
                to the scores.
        """

        #############################################################################
        # TODO: Compute for the softmax cross entropy loss                          #
        #############################################################################
        prob = self.softmax(scores)
        loss = np.sum(-np.log(prob[range(labels.shape[0]),labels]))/labels.shape[0]
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

        #############################################################################
        # TODO: Compute for the gradients of the loss with respect to the scores    #
        #############################################################################
        dloss = prob / labels.shape[0]
        losslist = (prob[range(labels.shape[0]),labels] - 1) / labels.shape[0]
        for i in range(labels.shape[0]):
            dloss[i][labels[i]] = losslist[i]
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

        return loss, dloss

    def network_forward(self, X):
        """
        This functions performs the forward pass which computes for the class scores given
        the input.

        Inputs:
        - X: A numpy array of shape (N, D) containing the data; there are N
          samples each of dimension D.
        
        Returns:
        scores: A numpy array of shape (N, C) containing class scores.
        cache_list: A list containing the cached values to be used on the backward pass.
        """
        scores = None
        cache_list = []
        #############################################################################
        # TODO: Perform a forward pass on the network and store the caches of       #
        # each layer inside the cache_list                                          #
        #############################################################################
        Layer1Value, cacheL1 = self.fully_connected_forward(X, self.params["W1"], self.params["b1"])

        if self.hidden_activation_fn == "sigmoid":
            Layer1ValueActivation, cacheL1A = self.sigmoid_forward(Layer1Value)
        elif self.hidden_activation_fn == "tanh":
            Layer1ValueActivation, cacheL1A = self.tanh_forward(Layer1Value)
        elif self.hidden_activation_fn == "relu":
            Layer1ValueActivation, cacheL1A = self.relu_forward(Layer1Value)

        scores, cacheL2 = self.fully_connected_forward(Layer1ValueActivation, self.params["W2"], self.params["b2"])

        # Cache
        cache_list =[cacheL1, cacheL1A, cacheL2]
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

        return scores, cache_list

    def network_backward(self, dloss, cache_list):
        """
        This functions performs the backward pass which computes for the gradients of the
        loss with respect to every parameter.

        Inputs:
        - dloss: A numpy array of shape (N, C) corresponding to the gradient of the 
            loss with respect to the scores outputted during the forward pass.
        - cache_list: A list of the cached values during the forward pass.
        
        Returns:
        grads: A dictionary containing the gradients of every parameter. For example, the gradients
            of the weights and bias of the first layer is stored in grads["W1"] and grads["b1"]
            respectively.
        """
        
        #############################################################################
        # TODO: Implement the backward pass.                                        #
        #############################################################################
        grads = {}
        X1, X1A, X2 = cache_list
        dX, grads["W2"], grads["b2"] = self.fully_connected_backward(dloss, X2)
        
        if self.hidden_activation_fn == "sigmoid":
            dX1A = self.sigmoid_backward(dX, X1A)
        elif self.hidden_activation_fn == "tanh":
            dX1A = self.tanh_backward(dX, X1A)
        elif self.hidden_activation_fn == "relu":
            dX1A = self.relu_backward(dX, X1A)
            
        _, grads["W1"], grads["b1"] = self.fully_connected_backward(dX1A, X1)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return grads

    def loss(self, X, y=None, lambda_reg=0.0):
        """
        Compute the loss and gradients for an iteration of linear regression.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the ground truth value for X[i].
        - reg: Regularization strength.

        Returns:
        Return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """

        # Unpack variables from the params dictionary
        N, D = X.shape

        # Compute the forward pass
        #############################################################################
        # TODO: Perform the forward pass, computing the class scores for the input. #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, C).                                                             #
        #############################################################################
        scores, cache_list = self.network_forward(X)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        # Compute the loss
        loss = None
        #############################################################################
        # TODO: Finish the forward pass, and compute the loss. This should include  #
        # both the data loss and L2 regularization for W1 and W2. Store the result  #
        # in the variable loss, which should be a scalar. Use the Softmax           #
        # classifier loss. So that your results match ours, multiply the            #
        # regularization loss by 0.5                                                #
        #############################################################################
        loss,dloss = self.softmax_cross_entropy_loss(scores,y)
        loss += 0.5 * lambda_reg * (np.sum(self.params['W1'] ** 2) + np.sum(self.params['W2'] ** 2))
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        # Backward pass: compute gradients. You should implement this function.
        grads = self.network_backward(dloss, cache_list)
        for i in range(self.num_layers):
            grads["W"+str(i+1)] += lambda_reg*self.params["W"+str(i+1)]
        #############################################################################
        # TODO: Compute the derivatives of the weights and biases. Store the        #
        # results in the grads dictionary. For example, grads['W'] should store     #
        # the gradient on W, and be a matrix of same size.The gradients should also #
        # include l2 the reguralizer.                                               #
        #############################################################################        
        pass

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return loss, grads

    def train_step(self, X, y, learning_rate=1e-3, lambda_reg=1e-5, batch_size=200):

        num_train, dim = X.shape

        indices = np.random.choice(num_train,batch_size, replace=False)
        X_batch = X[indices]
        y_batch = y[indices]

        loss, grads = self.loss(X_batch, y=y_batch, lambda_reg=lambda_reg)

        for i in range(self.num_layers):
            self.params["W"+str(i+1)] += - learning_rate * grads["W"+str(i+1)]
            self.params["b"+str(i+1)] += - learning_rate * grads["b"+str(i+1)]

        return loss, grads


    def train(self, X, y, learning_rate=1e-3, lambda_reg=0.0, num_iters=100, std_dev=1e-2,
            batch_size=200, verbose=False, one_step=False):
        """
        Train Linear Regression using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N, 1) containing the ground truth values.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        num_train, dim = X.shape
        self.initialize_weights(dim, std_dev)

        loss_history = []
        for it in range(num_iters):

            loss, grads = self.train_step(X, y, learning_rate, lambda_reg, batch_size)

            if it % 100 == 0:
                loss_history.append(np.squeeze(loss))

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return loss_history

    def predict(self, X, return_scores=False):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D)
        -return_scores: flag to return the scores with prediction
        Returns:
          - prediction: the predicted label
          - scores: scores from the network forward
        """
        
        scores, cache_list = self.network_forward(X)
        probs = self.softmax(scores)
        prediction = np.argmax(probs, axis=1)

        if return_scores:
            return prediction, scores
        else:
            return prediction

    def sigmoid_forward(self, x):
        """
        Computes the forward pass for a layer of sigmoid.

        Input:
        - x: Inputs, of any shape

        Returns a tuple of:
        - out: Output, of the same shape as x
        - cache: x
        """
        out = None
        #############################################################################
        # TODO: Implement the ReLU forward pass.                                    #
        #############################################################################
        out = 1/(1 + np.exp(-x))
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        cache = out
        return out, cache


    def sigmoid_backward(self, dout, cache):
        """
        Computes the backward pass for a layer of sigmoid.

        Input:
        - dout: Upstream derivatives, of any shape
        - cache: Input x, of same shape as dout

        Returns:
        - dx: Gradient with respect to x
        """
        out = cache
        #############################################################################
        # TODO: Implement the ReLU backward pass.                                   #
        #############################################################################
        dsigmoid = dout*(out*(1-out))
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return dsigmoid

    def tanh_forward(self, x):
        """
        Computes the forward pass for a layer of tanh.

        Input:
        - x: Inputs, of any shape

        Returns a tuple of:
        - out: Output, of the same shape as x
        - cache: x
        """
        out = None
        #############################################################################
        # TODO: Implement the ReLU forward pass.                                    #
        #############################################################################
        out = np.tanh(x)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        cache = out
        return out, cache


    def tanh_backward(self, dout, cache):
        """
        Computes the backward pass for a layer of tanh.

        Input:
        - dout: Upstream derivatives, of any shape
        - cache: Input x, of same shape as dout

        Returns:
        - dx: Gradient with respect to x
        """
        out = cache
        #############################################################################
        # TODO: Implement the ReLU backward pass.                                   #
        #############################################################################
        dtanh = dout*(1-out**2)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return dtanh


