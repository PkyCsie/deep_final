# coding=UTF-8
#!/usr/bin/env python
from gensim.models import Word2Vec
import numpy as np
def question_makeItEasyToUse(file_name,write_file_name,model):
    file = open(file_name,'r')
    write_file = open(write_file_name,'w')
    q_dict = {}
    file.readline()#fist row is not sentence
    for line in file:  
        vec_average = np.zeros(300)
        length = 0 
        sentence = line.split('\t')
        #print sentence
        question = sentence[2]
        question = question.split('\"')
        question = question[1].split('?')[0]
        for word in question.split(' '):
            try:
                vec_average += model[word]
                length += 1.0
            except: 
                try:
                    vec_average += model[word.split('\'')[0]]
                    length += 1.0
                except:         
                     try:
                         vec_average += model[word.split(',')[0]]
                         length += 1.0
                     except:
                        print word
        q_dict[sentence[1]] = vec_average
    for key in q_dict:
        print key
    file.close()
    write_file.close()


def choices_makeItEasyToUse(file_name,write_file_name,model):
    file = open(file_name,'r')
    write_file = open(write_file_name,'w')
    file.readline()#fist row is not sentence
    no = 0
    error = {}
    for line in file: 
        sentence = line.replace("\n",'').replace("(A)",'[').replace("(B)",'[').replace("(C)",'[').replace("(D)",'[').replace("(E)",'[').replace("\"",'')
        choice = sentence.split('\t')[2:]
        choice =  choice[0].split('[')[1:]
        for answer in choice:
           ave_ans_vec = np.zeros(300)
           length = 0.0
           for word in answer.split(' '):
                try:
                    ave_ans_vec += model[word]
                    length += 1.0
                except:
                    print word
                    error[word] = error.get(word,0) + 1
    for key in error:
        write_file.write(key+': '+str(error[key])+'\n')
        '''
        for choice in sentence:
            if choice!='' and choice.split(' ')[0] != ''and choice.split(' ')[0] != ' ':
                write_file.write(choice+'\n')
        '''
        #no += 1
        #assert no !=5,"no = 2"
    file.close()
    write_file.close()
if __name__ == '__main__':
    file_name = 'choices.train'
    write_file_name = 'where_is_not_key'
    model = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    #question_makeItEasyToUse(file_name,write_file_name,model)
    #file_name = 'choices.train'
    choices_makeItEasyToUse(file_name,write_file_name,model)
