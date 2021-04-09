import numpy as np

class Loss:
    def __init__(self, loss):
        self.loss = loss
        if (loss == 'binary_crossentropy'):
            self.forward_prop_fn = self.__binary_crossentropy__
            self.backward_prop_fn = self.__dbinary_crossentropy__
        elif (loss == 'categorical_crossentropy'):
            self.forward_prop_fn = self.__categorical_crossentropy_with_logits__
            self.backward_prop_fn = self.__dcategorical_crossentropy_with_logits__
        elif (loss == 'mse'):
            self.forward_prop_fn = self.__mse__
            self.backward_prop_fn = self.__dmse__
            
    def __mse__(self, y_true, y_pred):
        return np.mean(np.power(y_true-y_pred, 2))

    def __dmse__(self, y_true, y_pred):
        return 2*(y_pred-y_true)/y_true.size
    
    def __binary_crossentropy__(self, y_true, y_pred):
        return -(1 / len(y_true)) * ((y_true * np.log(y_pred)) + ((1 - y_true) * np.log(1 - y_pred)))

    def __dbinary_crossentropy__(self, y_true, y_pred):
        return -(np.divide(y_true, y_pred) - np.divide(1 - y_true, 1 - y_pred))
    
    def __categorical_crossentropy_with_logits__(self, y_true, z):
        ones = np.argwhere(y_true == 1)
        exp = np.exp(z - np.max(z, axis=1, keepdims=True))
        smax = exp / np.sum(exp, axis=1, keepdims=True)
        self.__value__ = smax
        return -np.mean(np.log(smax[range(y_true.shape[0]),y_true]))
    
    def __dcategorical_crossentropy_with_logits__(self, y_true, z):
        grads = self.__value__
        grads[range(y_true.shape[0]),y_true] -= 1
        grads /= y_true.shape[0]
        return grads