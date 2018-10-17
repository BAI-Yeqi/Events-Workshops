# -*- coding: utf-8 -*-

from base.base_model import BaseModel
import tensorflow as tf


class GRUModel(BaseModel):
    def __init__(self, config):
        super(GRUModel, self).__init__(config)
        
        # GRU specific parameters
        # num of steps per training example (number of time steps), we hope to make this a flexible value
        self.num_steps = config.num_steps
        # embedding size per item to be trained
        self.user_emb_size = config.user_emb_size
        self.item_emb_size = config.item_emb_size

        # size of w2v embedding from item description
        self.w2v_size = config.w2v_size
        # number of items in Shopee SG
        self.num_item = config.num_item
        self.num_user = config.num_user

        # measures the depth of GRUs we are using, by default is 1
        self.num_layers = config.num_layers
        # dropout value, by default 0
        self.dropout = config.dropout
        # batch size
        self.batch_size = config.batch_size
        # define the number of units used for the embedding GRU
        self.num_units_4emb = config.num_units_4emb
        
        
        self.build_model()
            
        self.init_saver()
        


    def build_model(self):
        with tf.device(self.config.GPU_name): 
            self.is_training = tf.placeholder(tf.bool)

            # the input should be batch_size * num_steps * one (item index) 
            # do note the generated training data should agree with the placeholder defined
            self.itemid_input = tf.placeholder(tf.int32, shape=[None, None])

            self.userid_input = tf.placeholder(tf.int32, shape=[None,])
            # input label, remember that for GRU4REC the negative training data are sampled
            self.y = tf.placeholder(tf.int32, shape=[None,])

            # random initialize the embedding layer (embedding matrix)
            item_embeddings = tf.Variable(
                tf.random_uniform([self.num_item, self.item_emb_size], -1.0, 1.0))
            # look up for the embedding for the specified item (to perform forward/back propagation)
            item_embed = tf.nn.embedding_lookup(item_embeddings, self.itemid_input)

            user_embeddings = tf.Variable(
                tf.random_uniform([self.num_user, self.user_emb_size], -1.0, 1.0))
            # look up for the embedding for the specified item (to perform forward/back propagation)
            user_embed = tf.nn.embedding_lookup(user_embeddings, self.userid_input)

            # first GRU model, based on embedding
            emb_cells = []
            for _ in range(self.num_layers):
                emb_cell = tf.contrib.rnn.GRUCell(self.num_units_4emb)  # Or LSTMCell(num_units)
                emb_cell = tf.contrib.rnn.DropoutWrapper(
                                              emb_cell, output_keep_prob=1.0 - self.dropout)
                emb_cells.append(emb_cell)
            emb_cell = tf.contrib.rnn.MultiRNNCell(emb_cells)

            emb_output, emb_state = tf.nn.dynamic_rnn(emb_cell, item_embed, dtype=tf.float32)
            emb_top_state = emb_state[-1]
            print("shape of emb_top_state: ", emb_top_state)
            concat_layer = tf.concat([emb_top_state, user_embed], 1)
            logits = tf.layers.dense(concat_layer, self.num_item, name="softmax")
            print("shape of the logits: ", logits.shape)
            self.prediction = tf.argmax(logits, axis = 1)
        
        with tf.name_scope("loss"):
            with tf.device(self.config.GPU_name): 
                self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, 
                                                                                    logits=logits)
                self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(
                                                        self.cross_entropy,
                                                        global_step=self.global_step_tensor)
                self.loss = tf.reduce_mean(self.cross_entropy, name="loss")
                # omit the optimizer at here because the optimization is in another module
                # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                # training_op = optimizer.minimize(loss)

            # self.correct: a Tensor Containing Booleans, 
            # with the shape of the array to be same as the batch_size
            self.correct = tf.nn.in_top_k(logits, self.y, 20)
            # the accuracy of the current batch, tf.cast is used for data type conversion
            self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))


    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
