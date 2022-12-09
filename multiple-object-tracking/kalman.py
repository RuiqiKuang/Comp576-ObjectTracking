import random
import time
import numpy as np
import const


class Kalman:
    def __init__(self, A, B, H, Q, R, X, P):
       
        self.A = A  
        self.B = B  
        self.H = H  
        self.Q = Q  
        self.R = R  
        self.X_posterior = X  # [center x,center y,w,h,dx,dy]
        self.P_posterior = P
        self.X_prior = None
        self.P_prior = None
        self.K = None
        self.Z = None  
      
        self.cache = const.cache
        
        self.track = []
        self.track_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.track.append([int(self.X_posterior[0]), int(self.X_posterior[1])])

    def predict(self):
        T1 = time.time()
        self.X_prior = np.dot(self.A, self.X_posterior)
        self.P_prior = np.dot(np.dot(self.A, self.P_posterior), self.A.T) + self.Q
        T2 = time.time()
        const.time_kalman += T2 - T1
        return self.X_prior, self.P_prior

    def update(self, det=None):
        """
       
        :param det:
        :return:
        """
        T3 = time.time()
        status = True
        if det is not None:  # 
            self.Z = det
            self.K = np.dot(np.dot(self.P_prior, self.H.T),
                            np.linalg.inv(np.dot(np.dot(self.H, self.P_prior), self.H.T) + self.R))
            self.X_posterior = self.X_prior + np.dot(self.K, self.Z - np.dot(self.H, self.X_prior))
            self.P_posterior = np.dot(np.eye(6) - np.dot(self.K, self.H), self.P_prior)
            status = True
        else:  # 
            if self.cache <= 1:
                status = False
            else:
                self.cache -= 1
                self.X_posterior = self.X_prior
                self.P_posterior = self.P_prior
                status = True
        if status:
            self.track.append([int(self.X_posterior[0]), int(self.X_posterior[1])])
        T4 = time.time()
        const.time_kalman += T4 - T3
        return status, self.X_posterior, self.P_posterior
