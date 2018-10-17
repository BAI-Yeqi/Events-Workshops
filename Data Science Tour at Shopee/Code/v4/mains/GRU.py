'''
This version of GRU enables to define the config as a pythn dictionary from another python fle or jupyter notebook

And pass in the dict as an argument to main function

'''


import tensorflow as tf

from data_loader.data_generator import DataGenerator
from models.GRU_model import GRUModel
from trainers.GRU_trainer import GRUTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger


def main(config_dict):
    # capture the config path from the run arguments
    # then process the json configuration file
 
    try:
        config = process_config(config_dict)
        

    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    # create tensorflow session
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # create your data generator
    data = DataGenerator(config)
    
    # create an instance of the model you want
    model = GRUModel(config)
    # create tensorboard logger
    logger = Logger(sess, config)
    # create trainer and pass all the previous components to it
    trainer = GRUTrainer(sess, model, data, config, logger)
    #load model if exists
    model.load(sess)
    # here you train your model
    trainer.train()


if __name__ == '__main__':
    main()
# -*- coding: utf-8 -*-

