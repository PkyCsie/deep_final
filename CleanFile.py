#!/usr/bin/env python

import numpy as np
def question_makeItEasyToUse(file_name,image_to_question,model):
    file = open(file_name,'r')
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
                        continue
        for question in image_to_question[sentence[0]]:
            if  question[0] == sentence[1]:
                   question.append(np.array([vec_average/length]))
                   break
    file.close()


def choices_makeItEasyToUse(file_name,image_to_question,model):
    file = open(file_name,'r')
    file.readline()#fist row is not sentence
    no = 0
    error = {}
    for line in file: 
        sentence = line.replace("\n",'').replace("(A)",'[').replace("(B)",'[').replace("(C)",'[').replace("(D)",'[').replace("(E)",'[').replace("\"",'')
        sentence = sentence.split('\t')
        choice = sentence[2:]
        question_id = sentence[1]
        choice =  choice[0].split('[')[1:]
        
        
        
        for question in image_to_question[sentence[0]]:
            if  question[0] == sentence[1]:
                for answer in choice:
                    ave_ans_vec = np.zeros(300)
                    length = 0.0
                    for word in answer.split(' '):
                        try:
                            ave_ans_vec += model[word]
                            length += 1.0
                        except:
                            #print word
                            continue
                    question.append(np.array([ave_ans_vec/length]))                
                question.append(question[question[1]+2]) #question id, which one is answer ,so answer = 1 + which one
    
    
        #no += 1
        #assert no !=5,"no = 2"
    file.close()
    
    
def creat_hash_table_image_to_question(file_name):    
    image_to_question = {}
    file = open(file_name,'r')
    file.readline()#fist row is not sentence
    for line in file: 
        sentence = line.replace("\n",'').split('\t')
        image_to_question[sentence[0]] = image_to_question.get(sentence[0],[])
        no = -1
        if sentence[2] == 'A':
           no = 1
        elif sentence[2] == 'B':
            no = 2
        elif sentence[2] == 'C':
            no = 3
        elif sentence[2] == 'D':
            no = 4
        elif sentence[2] == 'E':
            no = 5
        
        image_to_question[sentence[0]].append([sentence[1],no])
        #print image_to_question[sentence[0]]
    return image_to_question
'''  
if __name__ == '__main__':
    model = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
    image_to_question = creat_hash_table_image_to_question('answer.train_sol')
    question_makeItEasyToUse('question.train',image_to_question,model)
    choices_makeItEasyToUse('choices.train',image_to_question,model)
    print image_to_question
   
'''