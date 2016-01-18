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


word_vec_dim = 100
conv_dim = 192
concate_dim = word_vec_dim + conv_dim

question = tf.placeholder(tf.float32, [None, None])
answer = tf.placeholder(tf.float32, [None, 100])
choice1 = tf.placeholder(tf.float32, [None, 100])
choice2 = tf.placeholder(tf.float32, [None, 100])
choice3 = tf.placeholder(tf.float32, [None, 100])
choice4 = tf.placeholder(tf.float32, [None, 100])
conv_feature = tf.placeholder(tf.float32, [None,192])
conv_feature_2 = tf.reshape(conv_feature,[5041,-1])

W = tf.Variable(tf.random_normal([192, 192], 0, 0.1))
b = tf.Variable(tf.zeros([192]))

qw = tf.Variable(tf.random_normal([100, 192], 0, 0.1))

W1 = tf.Variable(tf.random_normal([192, 1], 0, 0.1))
b1 = tf.Variable(tf.zeros([1]))


W2 = tf.Variable(tf.random_normal([concate_dim, concate_dim], 0, 0.1))
b2 = tf.Variable(tf.zeros([concate_dim]))


W3 = tf.Variable(tf.random_normal([concate_dim, concate_dim], 0, 0.1))
b3 = tf.Variable(tf.zeros([concate_dim]))

W4 = tf.Variable(tf.random_normal([concate_dim, concate_dim], 0, 0.1))
b4 = tf.Variable(tf.zeros([concate_dim]))

W4 = tf.Variable(tf.random_normal([concate_dim, concate_dim], 0, 0.1))
b4 = tf.Variable(tf.zeros([concate_dim]))

W5 = tf.Variable(tf.random_normal([concate_dim, word_vec_dim], 0, 0.1))
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
cost1 = tf.reduce_mean(tf.square(out_layer - answer))
cost2 = tf.reduce_mean(tf.square(out_layer - choice1))   
cost3 = tf.reduce_mean(tf.square(out_layer - choice2))
cost4 = tf.reduce_mean(tf.square(out_layer - choice3))
cost5 = tf.reduce_mean(tf.square(out_layer - choice4))
       
       
train_step = tf.train.AdagradOptimizer(0.01).minimize(cost)
saver = tf.train.Saver()

if __name__ == '__main__':

    #image_to_question = creat_hash_table_image_to_question('answer.train_sol')
    train_image_path = "/home/shen/Downloads/deep_final/train2014"
    
    print 'start:'
    model = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    print 'finish reading model'
    image_to_question = CleanFile.creat_hash_table_image_to_question('answer.train_sol')
    CleanFile.question_makeItEasyToUse('question.train',image_to_question,model)
    CleanFile.choices_makeItEasyToUse('choices.train',image_to_question,model)
    
    print 'finish read question and choice'
    for key in image_to_question:
        print image_to_question[key][0][image_to_question[key][0][1] + 1]
        print image_to_question[key][0][len(image_to_question[key][0])-1]
    
        assert 2==1,'stop'
    
    dirs = os.listdir(train_image_path)
    image_list = []
   
    for file in dirs:
        image_list.append(train_image_path+'/'+file)
        if len(image_list) == 1:
            questions = 100*np.ones((1,100))
            answers = np.ones((1,100))
            otherChoice1 = np.random.rand(1,100)
            otherChoice2 = np.random.rand(1,100)
            otherChoice3 = np.random.rand(1,100)
            otherChoice4 = np.random.rand(1,100)
            conv_array = classify_image.run_inference_on_image(image_list)
            sess = tf.Session()
            #saver.restore(sess,'my-model-90')
            sess.run(tf.initialize_all_variables())
            for s in xrange (100):
                ss = sess.run([cost,cost1,cost2,cost3,cost4,cost5,train_step],feed_dict={conv_feature:np.array(conv_array),question:questions,answer:answers,\
                                                           choice1:otherChoice1,choice2:otherChoice2,choice3:otherChoice3,\
                                                           choice4:otherChoice4,})
                if s % 10 == 0:
                    saver.save(sess, 'my-model', global_step=s)
                print ss[0],ss[1],ss[2],ss[3],ss[4],ss[5]
            sess.close()
            print file
            break
    