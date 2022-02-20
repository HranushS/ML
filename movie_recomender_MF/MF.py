import numpy as np
import pandas as pd
class MF:

    def __init__(self, R, K=3, alpha=0.01, beta=0.0001, max_iter=30):
        
        """
        Arguments
        
        R (ndarray) - user-item rating matrix
        K (int)- number of latent dimensions
        alpha (float) - learning rate
        beta (float) - regularization rate
        
        """
        if isinstance(R,pd.core.frame.DataFrame):
            self.R = R.to_numpy()
        elif isinstance(R,np.ndarray): 
            self.R=R
        else:
            raise Exception("Unsupported parapeter type")
        self.num_users, self.num_items = R.shape
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter

    def train(self):
        
        # Initializing user and item latent feature matrices and the biases
        self.P = np.random.rand(self.num_users, self.K)
        self.Q = np.random.rand(self.num_items, self.K)
        self.b_u = np.zeros(self.num_users)#np.array([np.mean(self.R[np.where(self.R != 0)])]*self.num_users)#
        self.b_i = np.zeros(self.num_items)#np.array([np.mean(self.R[np.where(self.R != 0)])]*self.num_items)#

        for i in range(self.max_iter):
            self.sgd()

        return self.full_matrix()
    
    def sgd(self):
        
        for i in range(self.num_users):
            for j in range(self.num_items):
                if self.R[i][j]!=0.0:    #zero means unrated by user

                    prediction = self.single_prediction(i, j)
                    eij = self.R[i][j] - prediction

                    self.b_u[i] += self.alpha * 2 * (eij - self.beta * self.b_u[i])
                    self.b_i[j] += self.alpha * 2 * (eij - self.beta * self.b_i[j])
                    
                    self.P[i, :] += self.alpha * 2 * (eij * self.Q[j, :] - self.beta * self.P[i,:])
                    self.Q[j, :] += self.alpha * 2 * (eij * self.P[i, :] - self.beta * self.Q[j,:])

    def single_prediction(self, i, j):
      
        return  self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
        

    def full_matrix(self):
        #print(self.P.dot(self.Q.T))
        return np.round(self.b_u[:,np.newaxis] + self.b_i[np.newaxis:,] + self.P.dot(self.Q.T),2)




