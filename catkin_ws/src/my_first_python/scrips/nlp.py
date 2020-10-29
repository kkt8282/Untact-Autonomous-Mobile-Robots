#!/usr/bin/env python3
# -*- coding:UTF-8 -*-

# Respeaker Mic Array
from tuning import Tuning
import usb.core
import usb.util
import time

# ROS
import rospy
from std_msgs.msg import Int32

from soynlp.postagger import Dictionary
from soynlp.postagger import LRTemplateMatcher
from soynlp.postagger import LREvaluator
from soynlp.postagger import SimpleTagger
from soynlp.postagger import UnknowLRPostprocessor

from bayes import BayesianFilter
from konlpy.tag import Hannanum

import speech_recognition as sr
import pandas as pd
import numpy as np
from gtts import gTTS
from io import BytesIO
from urllib.request import urlopen
from bs4 import BeautifulSoup

import copy
import os
import argparse
import pickle
import pyaudio
import time
import pygame
import sys
import math
import webbrowser

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import large_vari
import corona_info
import sequence
import day
from data_utils import Vocabulary
from data_utils import load_data_interactive
from data_loader import prepare_sequence, prepare_char_sequence, prepare_lex_sequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from CNN_BiLSTM import CNNBiLSTM
from data_loader import get_loader
from sklearn.metrics import f1_score
from gensim.models import word2vec

dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
webbrowser.open('file:///home/hustar/catkin_ws/src/my_first_python/scripts/bot_face/default.html', new=0, autoraise=True)
bot_onoff_index = 1

intent_bf = BayesianFilter()
bf = BayesianFilter()
IntentList = ['위치','절차','번호','일정','확진자','정보','종료']
call_bot = ''
text = ''
data = pd.read_csv('./data/병원 질의 데이터베이스 - intent_ 절차.csv')
person_data = pd.read_csv('./data/병원 질의 데이터베이스 - intent_ 확진자 수.csv')
location_data = pd.read_csv('./data/위치_대답.csv')
han = Hannanum()
loop_flag = False
flag = False
cnt = True
cont = 0
tagger = SimpleTagger(LRTemplateMatcher(Dictionary(large_vari.pos_dict)), LREvaluator(), UnknowLRPostprocessor())

vocab_path='./data_in/vocab_ko_NER.pkl'
char_vocab_path='./data_in/char_vocab_ko_NER.pkl'
pos_vocab_path='./data_in/pos_vocab_ko_NER.pkl'
lex_dict_path='./data_in/lex_dict.pkl'
model_load_path='./data_in/cnn_bilstm_tagger-179-400_f1_0.8739_maxf1_0.8739_100_200_2.pkl'
num_layers=2
embed_size=100
hidden_size=200 
gpu_index=0

predict_NER_dict = {0: '<PAD>',
					1: '<START>',
					2: '<STOP>',
					3: 'B_LC',
					4: 'B_DT',
					5: 'B_OG',
					6: 'B_TI',
					7: 'B_PS',
					8: 'I',
					9: 'O'}

NER_idx_dic = {'<unk>': 0, 'LC': 1, 'DT': 2, 'OG': 3, 'TI': 4, 'PS': 5}	

pretrained_word2vec_file = './data_in/word2vec/ko_word2vec_' + str(embed_size) + '.model'
wv_model_ko = word2vec.Word2Vec.load(pretrained_word2vec_file)
word2vec_matrix = wv_model_ko.wv.syn0

# build vocab
with open(vocab_path, 'rb') as f:
	vocab = pickle.load(f)
with open(char_vocab_path, 'rb') as f:
	char_vocab = pickle.load(f)
with open(pos_vocab_path, 'rb') as f:
	pos_vocab = pickle.load(f)
with open(lex_dict_path, 'rb') as f:
	lex_dict = pickle.load(f)

# build models
cnn_bilstm_tagger = CNNBiLSTM(vocab_size=len(vocab),
										char_vocab_size=len(char_vocab),
										pos_vocab_size=len(pos_vocab),
										lex_ner_size=len(NER_idx_dic),
										embed_size=embed_size,
										hidden_size=hidden_size,
										num_layers=num_layers,
										word2vec=word2vec_matrix,
										num_classes=10)

cnn_bilstm_tagger.load_state_dict(torch.load(model_load_path, map_location=lambda storage, loc: storage))
if torch.cuda.is_available():
	cnn_bilstm_tagger.cuda(gpu_index)

cnn_bilstm_tagger.eval()


def to_np(x):
    return x.data.cpu().numpy()

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda(gpu_index)
    return Variable(x, volatile=volatile)


def preprocessing(x_text_batch, x_pos_batch, x_split_batch):
    x_text_char_item = []
    for x_word in x_text_batch[0]:
        x_char_item = []
        for x_char in x_word:
            x_char_item.append(x_char)
        x_text_char_item.append(x_char_item)
    x_text_char_batch = [x_text_char_item]

    x_idx_item = prepare_sequence(x_text_batch[0], vocab.word2idx)
    x_idx_char_item = prepare_char_sequence(x_text_char_batch[0], char_vocab.word2idx)
    x_pos_item = prepare_sequence(x_pos_batch[0], pos_vocab.word2idx)
    x_lex_item = prepare_lex_sequence(x_text_batch[0], lex_dict)

    x_idx_batch = [x_idx_item]
    x_idx_char_batch = [x_idx_char_item]
    x_pos_batch = [x_pos_item]
    x_lex_batch = [x_lex_item]


    max_word_len = int(np.amax([len(word_tokens) for word_tokens in x_idx_batch])) # ToDo: usually, np.mean can be applied
    batch_size = len(x_idx_batch)
    batch_words_len = [len(word_tokens) for word_tokens in x_idx_batch]
    batch_words_len = np.array(batch_words_len)

    # Padding procedure (word)
    padded_word_tokens_matrix = np.zeros((batch_size, max_word_len), dtype=np.int64)
    for i in range(padded_word_tokens_matrix.shape[0]):
        for j in range(padded_word_tokens_matrix.shape[1]):
            try:
                padded_word_tokens_matrix[i, j] = x_idx_batch[i][j]
            except IndexError:
                pass

    max_char_len = int(np.amax([len(char_tokens) for word_tokens in x_idx_char_batch for char_tokens in word_tokens]))
    if max_char_len < 5: # size of maximum filter of CNN
        max_char_len = 5
        
    # Padding procedure (char)
    padded_char_tokens_matrix = np.zeros((batch_size, max_word_len, max_char_len), dtype=np.int64)
    for i in range(padded_char_tokens_matrix.shape[0]):
        for j in range(padded_char_tokens_matrix.shape[1]):
            for k in range(padded_char_tokens_matrix.shape[1]):
                try:
                    padded_char_tokens_matrix[i, j, k] = x_idx_char_batch[i][j][k]
                except IndexError:
                    pass

    # Padding procedure (pos)
    padded_pos_tokens_matrix = np.zeros((batch_size, max_word_len), dtype=np.int64)
    for i in range(padded_pos_tokens_matrix.shape[0]):
        for j in range(padded_pos_tokens_matrix.shape[1]):
            try:
                padded_pos_tokens_matrix[i, j] = x_pos_batch[i][j]
            except IndexError:
                pass

    # Padding procedure (lex)
    padded_lex_tokens_matrix = np.zeros((batch_size, max_word_len, len(NER_idx_dic)))
    for i in range(padded_lex_tokens_matrix.shape[0]):
        for j in range(padded_lex_tokens_matrix.shape[1]):
            for k in range(padded_lex_tokens_matrix.shape[2]):
                try:
                    for x_lex in x_lex_batch[i][j]:
                        k = NER_idx_dic[x_lex]
                        padded_lex_tokens_matrix[i, j, k] = 1
                except IndexError:
                    pass

                
    x_text_batch = x_text_batch
    x_split_batch = x_split_batch
    padded_word_tokens_matrix = torch.from_numpy(padded_word_tokens_matrix)
    padded_char_tokens_matrix = torch.from_numpy(padded_char_tokens_matrix)
    padded_pos_tokens_matrix = torch.from_numpy(padded_pos_tokens_matrix)
    padded_lex_tokens_matrix = torch.from_numpy(padded_lex_tokens_matrix).float()
    lengths = batch_words_len

    return x_text_batch, x_split_batch, padded_word_tokens_matrix, padded_char_tokens_matrix, padded_pos_tokens_matrix, padded_lex_tokens_matrix, lengths

def parsing_seq2NER(argmax_predictions, x_text_batch):
    predict_NER_list = []
    predict_text_NER_result_batch = copy.deepcopy(x_text_batch[0]) #tuple ([],) -> return first list (batch_size == 1)
    for argmax_prediction_seq in argmax_predictions:
        predict_NER = []
        NER_B_flag = None # stop B
        prev_NER_token = None
        for i, argmax_prediction in enumerate(argmax_prediction_seq):
                now_NER_token = predict_NER_dict[argmax_prediction.cpu().data.numpy()[0]]
                predict_NER.append(now_NER_token)
                if now_NER_token in ['B_LC', 'B_DT', 'B_OG', 'B_TI', 'B_PS'] and NER_B_flag is None: # O B_LC
                    NER_B_flag = now_NER_token # start B
                    predict_text_NER_result_batch[i] = '<'+predict_text_NER_result_batch[i]
                    prev_NER_token = now_NER_token
                    if i == len(argmax_prediction_seq)-1:
                        predict_text_NER_result_batch[i] = predict_text_NER_result_batch[i]+':'+now_NER_token[-2:]+'>'

                elif now_NER_token in ['B_LC', 'B_DT', 'B_OG', 'B_TI', 'B_PS'] and NER_B_flag is not None: # O B_LC B_DT
                    predict_text_NER_result_batch[i-1] = predict_text_NER_result_batch[i-1]+':'+prev_NER_token[-2:]+'>'
                    predict_text_NER_result_batch[i] = '<' + predict_text_NER_result_batch[i]
                    prev_NER_token = now_NER_token
                    if i == len(argmax_prediction_seq)-1:
                        predict_text_NER_result_batch[i] = predict_text_NER_result_batch[i]+':'+now_NER_token[-2:]+'>'

                elif now_NER_token in ['I'] and NER_B_flag is not None:
                    if i == len(argmax_prediction_seq) - 1:
                        predict_text_NER_result_batch[i] = predict_text_NER_result_batch[i] + ':' + NER_B_flag[-2:] + '>'

                elif now_NER_token in ['O'] and NER_B_flag is not None: # O B_LC I O
                    predict_text_NER_result_batch[i-1] = predict_text_NER_result_batch[i-1] + ':' + prev_NER_token[-2:] + '>'
                    NER_B_flag = None # stop B
                    prev_NER_token = now_NER_token

        predict_NER_list.append(predict_NER)
    return predict_NER_list, predict_text_NER_result_batch

def generate_text_result(text_NER_result_batch, x_split_batch):
    prev_x_split = 0 
    text_string = ''
    for i, x_split in enumerate(x_split_batch[0]):
        if prev_x_split != x_split:
            text_string = text_string+' '+text_NER_result_batch[i]
            prev_x_split = x_split
        else:
            text_string = text_string +''+ text_NER_result_batch[i]
            prev_x_split = x_split
    return text_string


def NER_print(input_str):
    input_str.replace("  ", "")
    input_str = input_str.strip()
    
    x_text_batch, x_pos_batch, x_split_batch = load_data_interactive(input_str)
    x_text_batch, x_split_batch, padded_word_tokens_matrix, padded_char_tokens_matrix, padded_pos_tokens_matrix, padded_lex_tokens_matrix, lengths = preprocessing(x_text_batch, x_pos_batch, x_split_batch)
    
    # Test
    argmax_labels_list = []
    argmax_predictions_list = []


    padded_word_tokens_matrix = to_var(padded_word_tokens_matrix, volatile=True)
    padded_char_tokens_matrix = to_var(padded_char_tokens_matrix, volatile=True)
    padded_pos_tokens_matrix = to_var(padded_pos_tokens_matrix, volatile=True)
    padded_lex_tokens_matrix = to_var(padded_lex_tokens_matrix, volatile=True)


    predictions = cnn_bilstm_tagger.sample(padded_word_tokens_matrix, padded_char_tokens_matrix, padded_pos_tokens_matrix, padded_lex_tokens_matrix, lengths)
    
    max_predictions, argmax_predictions = predictions.max(2)

    if len(argmax_predictions.size()) != len(
        predictions.size()):  # Check that class dimension is reduced or not (API version issue, pytorch 0.1.12)
        max_predictions, argmax_predictions = predictions.max(2, keepdim=True)

    argmax_predictions_list.append(argmax_predictions)
    
    predict_NER_list, predict_text_NER_result_batch = parsing_seq2NER(argmax_predictions, x_text_batch)    
    
    origin_text_string = generate_text_result(x_text_batch[0], x_split_batch)
    predict_NER_text_string = generate_text_result(predict_text_NER_result_batch, x_split_batch)

    return predict_NER_text_string

def init_fuction():    
    pygame.init()
    
    if sys.version_info <= (2,7):
        reload(sys)
        sys.setdefaultencoding('utf-8')
    
    for idx in range(data.shape[0]):
        intent_bf.fit(data['Question'][idx],data['Intent'][idx])
        
    for idx in range(person_data.shape[0]):
        bf.fit(person_data['질문'][idx],person_data['분류'][idx])
            
    
def callback(recognizer, audio):
    global call_bot
    try:
        call_bot = recognizer.recognize_google(audio, language='ko')
        print("you said : " + call_bot)
    except sr.UnknownValueError:
        pass
        #TTS('알아 듣지 못했습니다. 다시 한 번 말해주세요.')
    except sr.RequestError as e:
        TTS('리퀘스트 에러입니다. 에러는 {0}입니다.'.format(e))

def background_call():
    global call_bot
    print('Speak plz : ')
    r = sr.Recognizer()
    m = sr.Microphone()
    with m as source:
        r.adjust_for_ambient_noise(source)
    stop_listening = r.listen_in_background(m, callback)
    
    for _ in range(50):
        time.sleep(0.1)
    
    stop_listening(wait_for_stop = False)

def STT():
    global text
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print('Speak Anything :')
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio, language='ko')
            print('You said : {}'.format(text))
            return text
        except:
            print('Sirri could not recignize your voice')
            return ''
        #return text
def TTS(text):
    webbrowser.open('file:///home/hustar/catkin_ws/src/my_first_python/scripts/bot_face/talk.html', new=0, autoraise=True)
    tts = gTTS(text=text, lang='ko')
    fp = BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    pygame.mixer.init()
    pygame.mixer.music.load(fp)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    webbrowser.open('file:///home/hustar/catkin_ws/src/my_first_python/scripts/bot_face/default.html', new=0, autoraise=True)

def talker(bot_onoff):
    try:
        pub = rospy.Publisher('mask_check', Int32, queue_size =10)
        rospy.init_node('talker', anonymous = True)
        if not rospy.is_shutdown():
            rospy.loginfo(bot_onoff)
            pub.publish(bot_onoff)
        #rate = rospy.Rate(10)
        #rate.sleep() 
    except rospy.ROSInterruptException:
        pass
        
#def callback(data):
#    rospy.loginfo(rospy.get_caller_id() + "robot status: %s", data.data)
#    if data == 0:

#def listener():
#    rospy.init_node('listener', anonymous=True)
#    rospy.Subscriber("robot_status", Int32, callback)
#    rospy.spin()

def nlp():
	global call_bot
	global flag
	while True:
		try:
			global bot_onoff_index
			if bot_onoff_index == 1:
				talker(1)
				bot_onoff_index = 0
			background_call()
			while '코비' in call_bot:
				text = ''
				if bot_onoff_index == 0:
					talker(0)
					if dev:
						Mic_tuning = Tuning(dev)
					#talker(math.radians(Mic_tuning.direction))
					print(Mic_tuning.direction)
					if Mic_tuning.direction <= 45:
						bot_angle = 14
					elif Mic_tuning.direction <= 90:
						bot_angle = 13
					elif Mic_tuning.direction <= 135:
						bot_angle = 12
					elif Mic_tuning.direction <= 180:
						bot_angle = 11
					elif Mic_tuning.direction <= 225:
						bot_angle = 18
					elif Mic_tuning.direction <= 270:
						bot_angle = 17
					elif Mic_tuning.direction <= 315:
						bot_angle = 16
					elif Mic_tuning.direction <= 360:
						bot_angle = 15
					talker(bot_angle)
					bot_onoff_index = 1
				if cnt:
					TTS('무엇이 궁금하신가요.')
					cnt = False
				text = STT()
				intent, score = intent_bf.predict(text)

				for l in IntentList:
					if intent == l:
						c = IntentList.index(l)
                
				text = text.replace(" ","")
				token_list = tagger.tag(text)
				NER_string = NER_print(text)
                        
				if c==0:                 
					for ele in token_list:
						if ele[1] == 'Adverb':
							keyword = ele[0]
							loop_flag = True
                    
					if loop_flag == True:
						for idx in range(location_data.shape[0]):
							if location_data['키워드'][idx].strip() == keyword.strip():
								TTS(location_data['대답'][idx])
								loop_flag = False
					else:
						TTS('정확한 위치를 말씀해 주십시오.')
                    
				elif c==1:                    
					res_msg = sequence.sequence(text)
					TTS(res_msg)
                    
				elif c==2:
                    
					html = urlopen("https://www.pohangsmh.co.kr/content/01intro/13_03.php")
					bsObj = BeautifulSoup(html, "html.parser")
					loc = bsObj.findAll("th", {"scope" : "row"})
					number = bsObj.findAll("td", {"class" : "left"})
                    
					for ele in token_list:
						if ele[1] == 'Adverb':
							keyword = ele[0]
							loop_flag = True
                            
					if loop_flag == True:
						for i in range(51):
							if loc[i].text.strip() == keyword.strip():
								TTS("{} 전화번호는 {}입니다".format(loc[i].text.strip(), number[i].text.strip()))
								loop_flag = False
								break
					else:
						TTS('잘못 인식하였습니다 다르게 질문하여 주세요')
                    

				elif c==3: # 일정
					name = 0
					if 'PS' in NER_string:
						indx1 = NER_string.index('<')
						indx2 = NER_string.index(':')
						loop_flag = True
                        
					if loop_flag == True:
						for i in range(len(day.lst)):
							if day.lst[i][1] == NER_string[indx1+1:indx2]:
								k = day.dic[day.lst[i][0]]
								name = i
								html = urlopen("https://www.pohangsmh.co.kr/content/03medi/01_0102.php?mp_idx=%d" %k)
								bsObj = BeautifulSoup(html, "html.parser")
								loop_flag = False
							else:
								cont += 1
					else:
						TTS('잘못 인식하였습니다 다르게 질문하여 주세요')
						continue
                        
					if cont == 91:
						TTS('잘못 인식하였습니다 다르게 질문하여 주세요')
						cont = 0
						continue

					check = bsObj.findAll("td")
					check2 = bsObj.findAll("caption")
					doc_list = []
					doc_index = 12
					doc_name_index = 0
                    
					while True:
						doctor = []
						for i in range(doc_index-12,doc_index):
							if i == doc_index-12:
								doctor.append(check2[doc_name_index].text)
								doc_name_index +=1
							doctor.append(check[i].text)
						doc_list.append(doctor)
						if len(check[doc_index+1].text) >= 2:
							break
						doc_index += 12

					for i in doc_list :
						for line in i :
							if day.lst[name][1] in line:
								person = i

					dayList = ['월','화','수','목','금','토','월','화','수','목','금','토']
					am = 0
					pm = 1
					dayCnt = 0
					dayCheck = [[],[]]

					for idx in range(1, 6):
						if person[idx] == '●':
							dayCheck[am].append(dayList[dayCnt])
						dayCnt += 1

					for idx in range(6, len(person)):
						if person[idx] == '●':
							dayCheck[pm].append(dayList[dayCnt])
						dayCnt += 1

					TTS("{} 과장님 진료일정은 오전은 {}요일, 오후는 {}요일 입니다".format(day.lst[name][1], ' '.join(dayCheck[am]), ' '.join(dayCheck[pm])))
					cont = 0
                    
				elif c==4:
					html = urlopen("https://search.naver.com/search.naver?sm=top_hty&fbm=1&ie=utf8&query=%EC%BD%94%EB%A1%9C%EB%82%98")
					bsObj = BeautifulSoup(html, "html.parser")
					total = bsObj.findAll("p", {"class" : "info_num"})
					new = bsObj.findAll("em", {"class" : "info_variation"})
					region_total = bsObj.findAll("td", {"class" : "align_center"})
					region_new = bsObj.findAll("td", {"class" : "align_center"})
                    
					if 'LC' in NER_string:
						indx1 = NER_string.index('<')
						indx2 = NER_string.index(':')
						region_text = NER_string[indx1+1:indx2]
						if region_text.strip() == '코로나'.strip():
							TTS('잘못 인식하였습니다 다르게 질문하여 주세요')
							continue
						loop_flag = True
						flag = True
					else:
						loop_flag = True
                    
					if loop_flag == True:
						if flag == False:
							TTS('현재 대한민국의 전체 확진자는 {} 명이고'.format(total[0].text))
							TTS('신규확진자는 {} 명입니다'.format(new[0].text))
							flag = True
							continue
						else:
							for i in range(40):
								if region_total[i].text.strip() == region_text:
									TTS('현재 {}지역의 전체 확진자는 {} 명이고'.format(region_text, region_total[i+1].text.strip()))
									break
							for i in range(100):
								if region_new[i].text.strip() == region_text:
									TTS('신규확진자는 {} 명입니다.'.format(region_new[i+2].text.strip()))
									break
							loop_flag = False
					flag = False
				elif c==5:
					tts_text = corona_info.corona_info(text)
					TTS(tts_text)
                    
				elif c==6:
					TTS('종료합니다.')
					break
                    
				else :
					TTS('다시 말씀해 주시겠어요?')
				text = ''
			cnt = True
			call_bot = ''
		except KeyboardInterrupt:
			print('Ctrl + C 중지 ')
			break

def main():
    global flag
    global call_bot
    global cnt
    global cont
    init_fuction()
    nlp()


if __name__ == '__main__':
    main()
