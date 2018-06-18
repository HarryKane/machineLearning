import pandas as pd
import numpy as np


class GaussianNB(object):

    def __init__(
        self,
        lambda_ls = 1.0
        ):
        self.lambda_ls = lambda_ls

        
    def fit(self,X_train,Y_train):
        self.gen_p_dict(X_train,Y_train)
        self.y = np.unique(Y_train)
        

    def gen_p_dict(self,X_train,Y_train):
        self.p_dict = {}
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        for i in np.unique(Y_train):
            Y_train_df = pd.DataFrame({'y':Y_train})
            self.p_dict['y_'+str(i)] = Y_train_df[Y_train_df['y']==i].shape[0] / Y_train_df.shape[0]
            for j in range(X_train.shape[1]):
                X_train_df = pd.DataFrame({'x':X_train[:,j],'y':Y_train})
                Gauss_list = [X_train_df['x'][X_train_df['y']==i].mean(),X_train_df['x'][X_train_df['y']==i].std()]
                self.p_dict['x_'+str(i)+str(j)] = Gauss_list
                
    #the probability density for the Gaussian distribution 
    def prob_gaussian(self,mu,sigma,x):
        return ( 1.0/(sigma * np.sqrt(2 * np.pi)) * \
                np.exp( - (x - mu)**2 / (2 * sigma**2)) )

    def test(self,X_test,Y_test):  
        predict_arr = self.predict(X_test)
            
        res = pd.DataFrame({'predict':predict_arr,'Y_test':Y_test})
        acc = res[res['predict']==res['Y_test']].shape[0] / res.shape[0]
        
        return acc 
 
    def predict(self,X_test):      
        predict_proba_arr = self.predict_proba(X_test)
        
        predict_arr = np.argmax(predict_proba_arr,axis=1)
        for i in range(predict_arr.shape[0]):
            predict_arr[i] = self.y[predict_arr[i]]
            
        return predict_arr
    
    
    
    def predict_proba(self,X_test):
        predict_proba_df = None
        X_test = np.array(X_test)
        for i,n in enumerate(self.y):
            X_test_1 = pd.DataFrame({'x_'+str(i)+str(j): X_test[:,j] for j in range(X_test.shape[1])})
            for col in X_test_1.columns:
                X_test_1[col] = X_test_1[col].apply(lambda x: 
                self.prob_gaussian(self.p_dict[col][0],self.p_dict[col][1],x))
            X_test_1['y_'+str(i)] = 1.0
            for col in X_test_1.columns:
                X_test_1['y_'+str(i)] *= X_test_1[col]
            X_test_1['y_'+str(i)] *= self.p_dict['y_'+str(i)]
            X_test_1 = X_test_1[['y_'+str(i)]]
            if n == 0:
                predict_proba_df = X_test_1
            else:
                predict_proba_df = pd.concat([predict_proba_df,X_test_1],axis=1)
                
        
        predict_proba_arr = np.array(predict_proba_df)
        for i in range(predict_proba_arr.shape[0]):
            predict_proba_arr[i,:] /= predict_proba_arr[i,:].sum()
        return predict_proba_arr
        
        



























