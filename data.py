
# coding: utf-8

# In[88]:


import numpy as np
import os
from sklearn.cross_validation import train_test_split


# In[101]:


class Data():
    
    def __init__(self,data_path,batch_size=120,load_directly=False,X_data_path="",Y_data_path="",train_proportion=None):
        if batch_size>10:
            self.batch_size=batch_size
        else:
            self.batch_size=50
        if not load_directly:
            if os.path.exists(data_path):
                self.process_data_and_write_to_disk(data_path)

            elif os.path.exists(os.path.join(os.getcwd(),data_path)):
                self.process_data_and_write_to_disk(os.path.join(os.getcwd(),data_path))
            else:
                print ("no such file exists...Returning without loading any Data")
        else:
            self.load_data(X_data_path,Y_data_path,train_proportion)
    
    def process_data_and_write_to_disk(self,data_path,train_proportion=None,file_prefix="data"):
        
        self.data_path=data_path
        data=np.load(self.data_path)
        self.data_size=len(data)
        print ("Data size= "+str(self.data_size))
        X,Y=np.expand_dims(data[0][0],axis=0),np.expand_dims(np.array(data[0][1]),axis=0) #expanding dims 
        for i in np.arange(1,self.data_size):
#             print (i)
            X=np.concatenate((X,np.array([data[i][0]])),axis=0)
            Y=np.concatenate((Y,np.array([data[i][1]])),axis=0)
            
        X=np.expand_dims(X,axis=3)#because current data only 2d
        np.save(file_prefix+"_X.npy",X)
        np.save(file_prefix+"_Y.npy",Y)
        X,Y=None,None
        self.load_data(file_prefix+"_X.npy",file_prefix+"_Y.npy",train_proportion)
        
    def load_data(self,X_data_path,Y_data_path,train_proportion):
        if os.path.exists(X_data_path) and os.path.exists(Y_data_path):
            X=np.load(X_data_path)
            Y=np.load(Y_data_path)
            self.X_train,self.X_test,self.Y_train,self.Y_test=train_test_split(X,Y,train_size=train_proportion)
            X,Y=None,None
        else:
            print ("failed to load data")
    def get_next_batch(self):
        train_size=self.X_train.shape[0]
        num_batches=int(train_size/self.batch_size)
        inds=np.arange(train_size)
        np.random.shuffle(inds)
        X_train,Y_train=self.X_train[inds],self.Y_train[inds]
        for i in np.arange(num_batches):
            yield X_train[(i*self.batch_size):(i*self.batch_size)+self.batch_size],Y_train[(i*self.batch_size):(i*self.batch_size)+self.batch_size]

