import numpy as np


import classify_image


import os,sys


import CleanFile

import tensorflow as tf
from gensim.models import Word2Vec

def _cosine_cost(y, y_):
    def norm(v):
        return tf.sqrt(tf.reduce_sum(tf.square(v), 1, keep_dims=True))

    y = y / norm(y)
    y_ = y_ / norm(y_)
    return -tf.reduce_mean(tf.matmul(y, y_, transpose_b=True))


word_vec_dim = 300
conv_dim = 192
concate_dim = word_vec_dim + conv_dim

question = tf.placeholder(tf.float32, [None, None])
answer = tf.placeholder(tf.float32, [None, None])
choice1 = tf.placeholder(tf.float32, [None, None])
choice2 = tf.placeholder(tf.float32, [None, None])
choice3 = tf.placeholder(tf.float32, [None, None])
choice4 = tf.placeholder(tf.float32, [None, None])
conv_feature = tf.placeholder(tf.float32, [None,conv_dim])
conv_feature_2 = tf.reshape(conv_feature,[5041,-1])

W = tf.Variable(tf.random_normal([conv_dim, conv_dim], 0, 0.1))
b = tf.Variable(tf.zeros([conv_dim]))

qw = tf.Variable(tf.random_normal([word_vec_dim, conv_dim], 0, 0.1))

W1 = tf.Variable(tf.random_normal([conv_dim, 1], 0, 0.1))
b1 = tf.Variable(tf.zeros([1]))


W2 = tf.Variable(tf.random_normal([concate_dim, concate_dim], 0, 0.01))
b2 = tf.Variable(tf.zeros([concate_dim]))


W3 = tf.Variable(tf.random_normal([concate_dim, concate_dim], 0, 0.01))
b3 = tf.Variable(tf.zeros([concate_dim]))

W4 = tf.Variable(tf.random_normal([concate_dim, concate_dim], 0, 0.01))
b4 = tf.Variable(tf.zeros([concate_dim]))

W4 = tf.Variable(tf.random_normal([concate_dim, concate_dim], 0, 0.01))
b4 = tf.Variable(tf.zeros([concate_dim]))

W5 = tf.Variable(tf.random_normal([concate_dim, word_vec_dim], 0, 0.01))
b5 = tf.Variable(tf.zeros([word_vec_dim]))


match_hidden_1 = tf.matmul(conv_feature, W) + tf.matmul(question,qw) + b  
match = tf.nn.softmax( tf.reshape(tf.matmul(match_hidden_1, W1) + b1,[-1,5041]))
con_v = tf.matmul(match,conv_feature_2)
concate_question_con_v = tf.concat(1,[con_v,question])

hidden_1 = tf.matmul(concate_question_con_v,W2) + b2 
hidden_2 = tf.matmul(hidden_1,W3) + b3 
hidden_3 = tf.matmul(hidden_2,W4) + b4 

out_layer = tf.matmul(hidden_3,W5) + b5 

cost = tf.reduce_mean(tf.square(out_layer - answer))
       
       
train_step = tf.train.AdagradOptimizer(0.01).minimize(cost)
saver = tf.train.Saver()

if __name__ == '__main__':

    train_image_path = "/home/shen/Downloads/deep_final/train2014"
    
    print 'start:'
    model = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    print 'finish reading model'
    image_to_question = CleanFile.creat_hash_table_image_to_question('answer.train_sol')
    CleanFile.question_makeItEasyToUse('question.train',image_to_question,model)
    CleanFile.choices_makeItEasyToUse('choices.train',image_to_question,model)
    '''
    print 'finish read question and choice'
    for key in image_to_question:
        print image_to_question[key][0][image_to_question[key][0][1] + 1]
        print image_to_question[key][0][len(image_to_question[key][0])-1]
    
        assert 2==1,'stop'
    '''
    dirs = os.listdir(train_image_path)
    image_list = []
    image_id = []
    for number in xrange(100):
        nos = 0
        for key in image_to_question:
            image_list.append(train_image_path+'/'+'COCO_train2014_000000'+key+'.jpg')
            image_id.append(key)
            if len(image_list) == 10:
                conv_array = classify_image.run_inference_on_image(image_list)
                sess = tf.Session()
                saver.restore(sess,'my-model-0')
                #sess.run(tf.initialize_all_variables())
                
                costs = 0
                for i in xrange(len(image_id)):
                    this_image_id = image_id[i]
                    for j in xrange(len(image_to_question[this_image_id])):
                        costs = sess.run([cost,train_step],feed_dict={conv_feature:conv_array[i],question:image_to_question[this_image_id][j][2],answer:image_to_question[this_image_id][j][-1]})[0]
                        if costs != costs:
                            print image_to_question[this_image_id][j][0]
                            print image_to_question[this_image_id][j][1]
                            print image_to_question[this_image_id][j][2 + image_to_question[this_image_id][j][1]]
                            print ss[0]
                            print image_to_question[this_image_id][j][-1]
                            print image_to_question[this_image_id][j][2]
                            assert 2==1,'error'                
               
                saver.save(sess, 'my-model', global_step=0)
                sess.close()
                image_list = []
                image_id = []
                print costs    
        if nos%1000 == 0:
            saver.save(sess, 'nos-model', global_step=nos)
                
        nos += 1        