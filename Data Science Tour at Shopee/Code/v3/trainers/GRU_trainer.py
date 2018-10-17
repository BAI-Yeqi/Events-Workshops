from base.base_train import BaseTrain
from trainers.GRU_tester import GRUTester
# progress displayer tqdm
from tqdm import tqdm
import numpy as np

from sklearn.metrics import roc_auc_score


class GRUTrainer(BaseTrain):
    # the data here is from the data_generator.py
    def __init__(self, sess, model, data, config, logger):
        super(GRUTrainer, self).__init__(sess, model, data, config,logger)
        self.tester = GRUTester(config, model, sess)      

    def train_epoch(self):
        loop = tqdm(range(self.config.num_iter_per_epoch))
        losses = []
        accs = []
        batch_no = 0
        for _ in loop:
            # print('batch no:', batch_no)
            loss, acc = self.train_step()
            losses.append(loss)
            accs.append(acc)
            batch_no = batch_no + 1
        loss = np.mean(losses)
        acc = np.mean(accs)        
        print("Loss: ", loss, "Accuracy: ", acc)
        test_loss, test_acc = self.tester.loss_accuracy()
        print("Test loss: ", test_loss, "Test accuracy: ", test_acc)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            'loss': loss,
            'acc': acc,
        }
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)

    def train_step(self):
        batch_userid_input, batch_itemid_input, batch_y = next(self.data.next_batch(self.config.batch_size))
        feed_dict = {self.model.userid_input: batch_userid_input, 
                     self.model.itemid_input: batch_itemid_input, 
                     self.model.y: batch_y, 
                     self.model.is_training: True}
        
        _, loss, acc = self.sess.run([self.model.train_step, self.model.cross_entropy, self.model.accuracy],
                                     feed_dict=feed_dict)
        return loss, acc
# -*- coding: utf-8 -*-

