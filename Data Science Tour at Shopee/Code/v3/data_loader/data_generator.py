import numpy as np
import pandas as pd

class DataGenerator:
    def __init__(self, config):
        self.config = config
        # load data here
        # in this fake data generator, the number of training data is simulated to be 100000000
        # the second digit in size tuple is the time step in the training data
        # here we suppose the time step to be one
        self.pro_arr = np.load(self.config.X_train_path)
        self.slice_table = self.analyze_data()

    def next_batch(self, batch_size):
        
        '''
        Input:
            1. batch_size as integer
            2. self.pro_arr as numpy array, with columns: 
                [0] userid as integer
                [1] pathid as integer
                [2] path as numpy array containing itemids as integers
                [3] session_length as integer = length_input + length_output = length_input + 1
                [4] label (output itemid) as integer
        Output:
            1. next_batch_userids as numpy array, with size: [batch_size, ] or [batch_size, num_time_step]
            2. next_batch_itemids as numpy array, with size: [batch_size, num_time_step]
            3. next_batch_y as numpy array, with size: [batch_size, ]
        '''
        # first step: use probability to random select a session length to generate the next batch
        m = self.pro_arr.shape[0]
        rand = np.random.randint(0,m)
        length = 0
        start_index = 0
        end_index = 0
        for i in range(len(self.slice_table)):
            if rand >= self.slice_table['start_index'][i] and rand < self.slice_table['end_index'][i]:
                length = self.slice_table['length'][i]
                start_index = self.slice_table['start_index'][i]
                end_index = self.slice_table['end_index'][i]
                break
        # print("generating next batch with session length t_step =", length)
        next_batchids = np.random.randint(start_index, end_index, batch_size)
        next_batch_userids = self.pro_arr[next_batchids][:,0]
        next_batch_itemids = np.array(self.pro_arr[next_batchids][:,2].tolist())
        next_batch_y = self.pro_arr[next_batchids][:,-1]
        
        yield next_batch_userids, next_batch_itemids, next_batch_y    
        
    def analyze_data(self):
        '''
        Input:
            1. self.pro_arr as numpy array
        Output: 
            1. the slicing table of pro_arr
        '''
        df = pd.DataFrame(self.pro_arr)
        df.columns = ['userid', 'pathid', 'path', 'length', 'Y']
    
        print("Sample of training data: ")
        print(df.head())
    
        summary = df.groupby('length').count()
        summary = pd.DataFrame(summary.drop(columns=['pathid','path']))
        summary = summary.reset_index()
        summary = summary.drop(['Y'],axis = 1)
        summary.columns = ['length', 'count']
    
        slice_table = summary
        slice_table['start_index'] = slice_table['count']
        slice_table['end_index'] = slice_table['count']
    
        # here is the logic for create the start index + end index
        for i in range(1, len(slice_table)):
            slice_table['end_index'][i] = slice_table['end_index'][i-1] + slice_table['count'][i]
    
        for i in range(1, len(slice_table)):
            slice_table['start_index'][i] = slice_table['end_index'][i-1]
    
        slice_table['start_index'][0] = 0
        print("Here is the slicing table of the data: ")
        print(slice_table)
        
        return slice_table
        
