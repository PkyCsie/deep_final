import numpy as np


import classify_image


import os,sys


import CleanFile



if __name__ == '__main__':

    image_to_question = creat_hash_table_image_to_question('answer.train_sol')
    train_image_path = "/home/shen/Downloads/deep_final/train2014"
    
    
    dirs = os.listdir(train_image_path)
    for file in dirs:
        classify_image.run_inference_on_image(train_image_path+'/'+file)
        print file
    