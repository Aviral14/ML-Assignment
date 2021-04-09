import numpy as np
from tqdm.notebook import tqdm
from .Layer import Layer
from .Loss import Loss

class Model:
    def __init__(self, loss):
        self.layers = []
        self.loss = Loss(loss)

    def add(self, layer):
        self.layers.append(layer)
        
    def evaluate(self, X, y_true, batch_size=32):
        m = len(X)
        err = 0
        results = []
        for batch in range(0, m, batch_size):
            y_preds = X[batch:batch+batch_size]
            for layer in self.layers:
                y_preds = layer.forward_propagation(y_preds)
            results.append(y_preds)
            err += self.loss.forward_prop_fn(y_true[batch:batch+batch_size], y_preds)
        y_preds = np.vstack(results)
        preds_max = np.argmax(y_preds, axis=1)
        return np.mean(preds_max == y_true), err / m

    def predict(self, input_data, batch_size=32):
        m = len(input_data)
        results = []
        for batch in range(0, m, batch_size):
            preds = input_data[batch:batch+batch_size]
            for layer in self.layers:
                preds = layer.forward_propagation(preds)
            results.append(preds)
        return np.vstack(results)

    def fit(self, X, y, epochs, lr, batch_size=32):
        m = len(X)
        loss_hist = []
        acc_hist = []
        
        pbar = tqdm(range(epochs))
        for i in pbar:
            err = 0
            for batch in range(0, m, batch_size):
                preds = X[batch:batch+batch_size]
                for layer in self.layers:
                    preds = layer.forward_propagation(preds)

                err += self.loss.forward_prop_fn(y[batch:batch+batch_size], preds)

                grads = self.loss.backward_prop_fn(y[batch:batch+batch_size], preds)
                for layer in reversed(self.layers):
                    grads = layer.backward_propagation(grads, lr)

            err /= m
            loss_hist.append(err)
            acc, err = self.evaluate(X, y)
            acc_hist.append(acc)
            pbar.set_postfix({'Epoch': f'{i + 1}/{epochs}', 'Error': err, 'Accuracy': acc})
        return {'loss': loss_hist, 'accuracy': acc_hist}