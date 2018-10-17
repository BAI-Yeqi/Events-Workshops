import numpy as np
import pandas as pd

class GRUTester:
    def __init__(self, config, model, sess):
        self.config = config
        # load data here
        # in this fake data generator, the number of training data is simulated to be 100000000
        # the second digit in size tuple is the time step in the training data
        # here we suppose the time step to be one
        self.pro_arr = np.load(self.config.X_test_path)
        self.slice_table = self.analyze_data()
        self.model = model
        self.sess = sess
        self.test_list_userids, self.test_list_itemids, self.test_list_y = self.bucket_data()
        
    def loss_accuracy(self):
        # use the model to reflect the accuracy
        sum_loss = 0.0
        sum_acc = 0.0
        for i in range(len(self.test_list_y)):
            userid_input = self.test_list_userids[i]
            itemid_input = self.test_list_itemids[i]
            y = self.test_list_y[i]
            loss, acc = self.batch_loss_accuracy(200, userid_input, itemid_input, y)
            sum_loss = sum_loss + loss * len(y)
            sum_acc = sum_acc + acc * len(y)
        avg_loss = sum_loss / self.pro_arr.shape[0]
        avg_acc = sum_acc / self.pro_arr.shape[0]
        return avg_loss, avg_acc
    
    def batch_loss_accuracy(self, test_batch_size, userid_input, itemid_input, y):
        losses = []
        sum_acc = 0.0
        for i in range(len(y)//test_batch_size):
            batch_userid_input = userid_input[i*test_batch_size:(i+1)*test_batch_size]
            batch_itemid_input = itemid_input[i*test_batch_size:(i+1)*test_batch_size]
            batch_y = y[i*test_batch_size:(i+1)*test_batch_size]
            feed_dict = {self.model.userid_input: batch_userid_input, 
                         self.model.itemid_input: batch_itemid_input, 
                         self.model.y: batch_y, 
                         self.model.is_training: False}
            loss, acc = self.sess.run([self.model.cross_entropy, self.model.accuracy],
                                         feed_dict=feed_dict)
            losses = losses + list(loss)
            print('accuray of a test batch:', acc)
            sum_acc = sum_acc + acc*len(batch_y)
            
        # last batch
        i = len(y)//test_batch_size
        batch_userid_input = userid_input[(i)*test_batch_size:]
        batch_itemid_input = itemid_input[(i)*test_batch_size:]
        batch_y = y[(i)*test_batch_size:]
        feed_dict = {self.model.userid_input: batch_userid_input, 
                     self.model.itemid_input: batch_itemid_input, 
                     self.model.y: batch_y, 
                     self.model.is_training: False}
        loss, acc = self.sess.run([self.model.cross_entropy, self.model.accuracy],
                                         feed_dict=feed_dict)
        losses = losses + list(loss)
        sum_acc = sum_acc + acc*(len(y)%test_batch_size)

        avg_loss = np.mean(losses)
        avg_acc = sum_acc/len(y)
        
        return avg_loss, avg_acc
        
    def bucket_data(self):
        test_list_userids = []
        test_list_itemids = []
        test_list_y = []  
        for i in range(len(self.slice_table)):
            start_index = self.slice_table['start_index'][i] 
            end_index = self.slice_table['end_index'][i]
            next_batch_userids = self.pro_arr[start_index:end_index][:,0]
            next_batch_itemids = np.array(self.pro_arr[start_index:end_index][:,2].tolist())
            next_batch_y = self.pro_arr[start_index:end_index][:,-1]
            test_list_userids.append(next_batch_userids)
            test_list_itemids.append(next_batch_itemids)
            test_list_y.append(next_batch_y)
            # print("test_list_y: ", np.array(test_list_y).shape, test_list_y)
        
        return test_list_userids, test_list_itemids, test_list_y    
        
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
        print("Here is the slicing table of the test data: ")
        print(slice_table)
        
        return slice_table
        
