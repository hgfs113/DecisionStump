import numpy as np


class MoiseevDecisionPen:
    def __init__(self):
        self.dict = {}

    def fit(self, X, y):
        labels = set(y)
        
        for label in labels:
            X_c = X[y == label]
            mu = np.mean(X_c, axis=0)
            std = np.std(X_c, axis=0)
            self.dict[label] = {'mu':mu, 'sigma':std}
        return self

    def predict(self, X):
        N, d = X.shape
        score = np.zeros((N, len(self.dict)))
        
        idx2label = {}
        for i, (cls, params) in enumerate(self.dict.items()):
            idx2label[i] = cls
            score[:, i] = -0.5 * np.sum(np.log(params['sigma'])) \
                - ((X - params['mu']) @ np.diag(params['sigma']**-1) * (X - params['mu'])).sum(1)
        preds = np.array([idx2label[p] for p in np.argmax(score, axis=1)])
        return preds

