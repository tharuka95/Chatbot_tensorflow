# Chatbot 

#libraries

import numpy as np
import tensorflow as tf
import re
import time

### Data Processing ###
lines =open('movie_lines.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
conversations =open('movie_conversations.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')

#create dictionary to map
id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line)== 5:
        id2line[_line[0]] = _line[4]
        
        
### creating list  of coversations
conver_ids = []
for conver in conversations[:-1]:
    _conver = conver.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
    conver_ids.append(_conver.split(','))
    
#Getting separately  Q and Ans
questions =[]
answers = []
for conv_id in conver_ids:
    for i in range(len(conv_id)-1):
        questions.append(id2line[conv_id[i]])
        answers.append(id2line[conv_id[i+1]])


#Cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am",text)
    text = re.sub(r"he's", "he is",text)
    text = re.sub(r"she's", "she is",text)
    text = re.sub(r"that's", "that is",text)
    text = re.sub(r"what's", "what is",text)
    text = re.sub(r"where's", "where is",text)
    text = re.sub(r"\'ll", " will",text)
    text = re.sub(r"\'ve", " have",text)
    text = re.sub(r"\'re", " are",text)
    text = re.sub(r"\'d", " would",text)
    text = re.sub(r"won't", " will not",text)
    text = re.sub(r"can't", " cannot",text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "",text)
    return text

#clean Q's
    
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))


#clean A's
clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))


#creating a dict to map each word to its no of occurences
word2count ={}
for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
            
for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

#thresholding 
            
threshold = 20
questionswords2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold:
        questionswords2int[word] = word_number
        word_number += 1
        
answerswords2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold:
        answerswords2int[word] = word_number
        word_number += 1
        
 #add the last tokens    
 
 tokens =['<PAD>', '<EOS>', '<OUT>', '<SOS>']
 for token in tokens:
     questionswords2int[token]= len(questionswords2int)+1
 for token in tokens:
    answerswords2int[token]= len(answerswords2int)+1
 
#creating inverse dict to answerswords2int dict
    
answersint2word = {w_i: w for w, w_i in answerswords2int.items()}

#Add the EOS to every ans

for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>' 

#translate all the Q's and Ans to intergers and 
#replace all wordsthat were filtered out by<OUT>

questions_to_int =[]
for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questionswords2int:
            ints.append(questionswords2int['<OUT>'])
        else:
            ints.append(questionswords2int[word])
    questions_to_int.append(ints)
             
answers_to_int =[]
for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in answerswords2int:
            ints.append(answerswords2int['<OUT>'])
        else:
            ints.append(answerswords2int[word])
    answers_to_int.append(ints)
    
#sorting questions and answers by length of questions

sorted_clean_questions = []
sorted_clean_answers = []
for length in range(1, 25 + 1):
    for i in enumerate(questions_to_int):
        if len(i[1]) == length:
            sorted_clean_questions.append(questions_to_int[i[0]])
            sorted_clean_answers.append(answers_to_int[i[0]])


################### SEQ2SEQ MODEL #########################
            
            
#creating placeholders for the input and the targets
            
def model_input():
    inputs =tf.placeholder(tf.int32, [None, None], name= 'input')
    targets =tf.placeholder(tf.int32, [None, None], name= 'target')
    lr =tf.placeholder(tf.float32, name= 'learning_rate')
    keep_prob =tf.placeholder(tf.float32, name= 'keep_prob')
    return inputs,targets,lr,keep_prob

#preprocessing the targets

def preprocess_targets(targets, word2int, batch_size):
    left_side = tf.fill([batch_size, 1], word2int['<SOS>'])
    right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1])
    preprocessed_targets = tf.concat([left_side, right_side], 1)
    return preprocessed_targets

#Creating the Encoder RNN Layer
    
def encoder_rnn_layer(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    _, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
                                                      cell_bw = encoder_cell,
                                                      sequence_length = sequence_lenght,
                                                      inputs = rnn_inputs,
                                                      dtype = tf.float32)
    return encoder_state



#Decoding the training set
def decode_training_set

































  

















