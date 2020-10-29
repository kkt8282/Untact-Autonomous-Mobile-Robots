#!/usr/bin/env python3
# -*- coding:UTF-8 -*-

# ROS
import rospy
from std_msgs.msg import Int32

import speech_recognition as sr
import pandas as pd
import numpy as np
from gtts import gTTS

import copy
import os
import argparse
import pickle
import pyaudio
import time
import sys
import math
import webbrowser

from ctypes import *
import random
import os
import cv2
import time
import darknet
import argparse
import wave
from multiprocessing import Process
from ctypes import Structure, c_double
import threading
from threading import Thread, enumerate
from queue import Queue
from itertools import combinations
import pyaudio
import pyrealsense2 as rs
import numpy as np
from gtts import gTTS
import gc


webbrowser.open('file:///home/hustar/catkin_ws/src/my_first_python/scripts/bot_face/default.html', new=0, autoraise=True)
bot_onoff_index = 1
robot_status_index = 1
nlp_status = 0

def parser():
    """	
    * parser()#000000
        - yolov4 구동에 필요한 파일들의 경로를 default로 둬서 바로 가져와서 실행할 수 있도록 만듦.
        - 인자값을 받을 수 있는 인스턴스(객체) 생성 argparse.ArgumentParser()
        - parameter: description = 인자 도움말 전에 표시할 텍스트
    """

    """ ====================================================================================

    * argparse
    : python script를 터미널에서 실행할 때 명령어 옵션에 대한 parameter를 python으로 전달할 때 사용하는 방
        사용예 : $./argv_test.py [param1][param2]
                parameter를 입력하지 않으면 default가 출력

    ===================================================================================="""

    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    """
        입력받을 인자값 등록 parser.add_argument()
        parameter:
        - name or flags : 옵션 문자역의 이름이나 리스트. --input
        - type : 명령행 인자가 변환되어야 할 형
        - default : 인자가 명령행에 없는 경우 생성되는 값
        - help : 인자가 하는 일에 대한 간단한 설명
    """
    parser.add_argument("--input", type=str, default=0,
                        help="video source. If empty, uses webcam 0 stream")
    parser.add_argument("--out_filename", type=str, default="",
                        help="inference video name. Not saved if empty")
    parser.add_argument("--weights", default="/media/hustar/Hustar/darknet/mask_set/yolov4-obj_best.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--config_file", default="/media/hustar/Hustar/darknet/mask_set/yolov4-obj.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="/media/hustar/Hustar/darknet/mask_set/obj.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with confidence below this value")
    """
        명령창에서 주어진 인자를 파싱한다. parse_args()
        이 후 parser()로 인자값을 받아서 사용할 수 있다.
    """
    return parser.parse_args()



def str2int(video_path):
    """
    * str2int()
        - 받아오는 string type의 video_path를 int로 변환
        - ValueError(데이터 타입이 맞지 않을 때 발생하는 에러) 시 그대로 return
    """
    try:
        return int(video_path)
    except ValueError:
        return video_path



def check_arguments_errors(args):
    """
    * check_arguments_errors()
        - def parse()를 args로 받아와서 error를 검사
        - os.path.exists()로 받아오는 argument의 경로에 해당하는 파일이 존재하는를지 검사
        - 없다면 raise문으로 들어가서 예외처리로 들어감
    """

    """ ====================================================================================
    * assert 조건, '메세지' 
        : 가정 설정문 , assert 뒤에 조건에 만족하지 않으면 AssertionError 출력
    * raise 예외객체(예외내용)
        : 강제 예외처리 방법
    ===================================================================================="""

    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise (ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise (ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise (ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if str2int(args.input) == str and not os.path.exists(args.input):
        raise (ValueError("Invalid video path {}".format(os.path.abspath(args.input))))



def set_saved_video(input_video, output_video, size):
    """
        여기서 동영상 파일명,프레임,영상의 크기등을 지정하여
        video.write(image) 코드로 image 프레임을 저장.
    """
    """ ====================================================================================
    * cv2.VideoWriter_fourcc(*"코덱")
        : 디지털 미디어 포맷 코드를 생성, 인코딩 방식을 설정. 여기선 MJPG

    * cv2.CAP_PROP_FPS
        : 현재 프레임 개수

    * cv2.VideoWriter(outfile, fourcc, frame, size)
        : 영상을 저장하기 위한 object
          저장될 파일명, Codec정보(cv2.VideoWriter_fourcc), 초당 저장될 frame, 저장될 사이즈(), 컬러 저장 여부
    ===================================================================================="""

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    # fps = int(input_video.get(cv2.CAP_PROP_FPS))
    video = cv2.VideoWriter(output_video, fourcc, 30, (640, 480), 1)
    return video



""" ====================================================================================
* 
===================================================================================="""
def video_capture(frame_queue, darknet_image_queue, depth_queue, distance_queue):
	global nlp_status
	global cap
	global pipeline
	global align
	while True:
		cap = pipeline.wait_for_frames()
		aligned_frames = align.process(cap)
		depth_frame = aligned_frames.get_depth_frame().as_depth_frame()
		color_frame = aligned_frames.get_color_frame()
		if not depth_frame or not color_frame:
			continue
		distance_queue.put(depth_frame)
		frame = np.asanyarray(color_frame.get_data())
			
		depth_image = np.asanyarray(depth_frame.get_data())
		depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
			
		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			
		frame_resized = cv2.resize(frame_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
		depth_resized = cv2.resize(depth_colormap, (width, height), interpolation=cv2.INTER_LINEAR)
			
		frame_queue.put(frame_resized)
		depth_queue.put(depth_resized)
		darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
		darknet_image_queue.put(darknet_image)
	cap.release()

def prioriry_tts(tts_set):
    global speaking_state2
    global speaking_state
    file_path_distance = './save_mp3_file/2m_output.wav'
    file_path = './save_mp3_file/output.wav'
    distance_thread = threading.Thread(target=tts_speak2, args=(file_path_distance,))
    mask_thread = threading.Thread(target=tts_speak, args=(file_path,))
    mask_thread.daemon = True
    distance_thread.daemon = True

    for i in range(len(tts_set)):
        temp = tts_set.pop()
        if temp == 1:
            mask_thread.start()
            mask_thread.join(timeout=20000)
        elif temp == 2:
            distance_thread.start()
            distance_thread.join(timeout=20000)
    speaking_state = False
    speaking_state2 = False

def inference(darknet_image_queue, detections_queue, fps_queue, distance_queue):
	global nlp_status
	while True:
		global drawing_queue
		global tts_race
		global tts_distance
		global speaking_state
		global speaking_state2
		global red_distance_queue
		global input_data
		objectId = 0
		input_data = 1
		search_depth = []
		tts_race = False
		tts_distance = False
		red_centroid = []

		'frame_image를 bytes array 형태로 만들어서 queue에 넣어둔 image를 가져옴'
		darknet_image = darknet_image_queue.get()

		prev_time = time.time()
		"""
			detection = label, confidence, (x,y,w,h)
		"""
		detections = darknet.detect_image(network, class_names, darknet_image, thresh=args.thresh)
		detections_mask = darknet.detect_image(network_mask, class_names_mask, darknet_image, thresh=args.thresh)
		detections.extend(detections_mask)
		detections_queue.put(detections)
		'distance_obj = darknet.detect_image에서 가져온 detections'
		distance_obj = distance_queue.get()

		fps = int(1 / (time.time() - prev_time))
		fps_queue.put(fps)
		'detection된 box안의 centroid x값을 기준으로 정렬 -> frame 기준으로 왼쪽부터 정렬됨.'
		detections.sort(key=lambda e: e[2][0])
		for i in detections:
			'detection된 '
			#if 'person' in i[0] and speaking_state == False:
			if 'no_mask' in i[0] and not speaking_state:
				tts_race = True
			'accuracy가 80% 이상일 '
			if 'person' in i[0] and float(i[1]) >= 60:
				x = i[2][0]
				y = i[2][1]
				w = i[2][2]
				h = i[2][3]
				depth_sum = []
				for j in range(int(y-i[2][3]//5) + 1, int(y+i[2][3]//5)-1):
				#     rand_loc = random.randint(x - int(w // 5), x + int(w // 5))
					if 0 < x < 640 and 0 < y < 480:
							#print(rand_loc, j)
						depth_x = int(x * 2.5)
						depth_y = int(y * 2.5)
						depth = distance_obj.get_distance(depth_x, depth_y)
						depth_sum.append(depth)
				depth_sum = np.array(depth_sum)
				median_depth = round(np.median(depth_sum), 3)
				objectId += 1
					#print(median_depth) # 객체까지의 실제거리(사람까지의 거리)
				if depth >= 1.0:
					search_depth.append([int(x), int(y), median_depth, objectId, w, h]) # x,y,z, ObjectId,height
				elif depth < 1.0:
					input_data = 0 # ros message data, 사람이 1미터 이내 일때, 메세지 보내서 멈춤
						#continue
			#print("input_data : ", input_data)
			#if input_data == 0: # 연산량 줄이기 위함, 건너뛰기로 인해 조합연산하지 않고 넘어간다
				#continue
			#if input_data == 1:
		drawing_queue = search_depth
		for (x1, y1, z1, ObjectId1, width1, height1), (x2, y2, z2, ObjectId2, width2, height2) in combinations(search_depth, 2):
			real_distance = calcul_distance((x1, y1, z1, ObjectId1), (x2, y2, z2, ObjectId2))
			print("distance", round(real_distance, 3))
			if real_distance <= 1.5:
				red_centroid.append([(x1, y1, ObjectId1, width1, height1), (x2, y2, ObjectId2, width2, height2)])
				#if real_distance <= 1.3 and speaking_state2 == False and tts_distance == False:
			if real_distance <= 1.3 and not speaking_state2 and not tts_distance:
				tts_distance = True
		red_distance_queue = red_centroid
		try:
			tts_set = []

			if tts_race and not speaking_state :
			#if tts_race == True and speaking_state == False:
				tts_set.append(1)

			if tts_distance and not speaking_state2:
#            if tts_distance == True and speaking_state2 == False :
				tts_set.append(2)
			if not speaking_state and not speaking_state2:
				tts_priority_set = threading.Thread(target=prioriry_tts, args=(tts_set,))
				tts_priority_set.daemon = True
				tts_priority_set.start()
		except Exception:
			continue
			#print("FPS: {}".format(fps))
			#darknet.print_detections(detections, args.ext_output)

	cap.release()

def return_input():
    global input_data
    return int(input_data)


def calcul_distance(p1, p2):
    """
        p1, p2의 Realpoint를 계산한 뒤, 벡터의 거리 계산하는 공식으로 Real_distance를 계산
        여기서 단위는 이미 미터로 계산
    """
    Real_p1 = calcul_realpoint(p1)
    Real_p2 = calcul_realpoint(p2)
    Real_distance = np.sqrt((Real_p1[0]-Real_p2[0])**2 + (Real_p1[1]-Real_p2[1])**2 + (Real_p1[2]-Real_p2[2])**2)
    return Real_distance


def calcul_realpoint(p1):
    """
        image상의 좌표를 real 좌표로 변환하는 함수
        Real_X = ((출력된 영상 상의 X좌표) - (영상의 중앙 X좌표)) * (pixel pitch(m)) / (focal length(m))
        Real_Y = ((출력된 영상 상의 Y좌표) - (영상의 중앙 Y좌표)) * (pixel pitch(m)) / (focal length(m))
        pixel pitch : 3.0마이크로 미터
        focal length : 1.93밀리미터
        미터로 단위를 맞춰서 real 좌표를 계산
    """
    Real_X = ((((p1[0] - 320) * 0.000003) / 0.00193) * p1[2])
    Real_Y = -((((p1[1] - 240) * 0.000003) / 0.00193) * p1[2])
    Real_Z = p1[2]
    # print("p", [p1[0], p1[1], p1[2]])
    # print("Real_point", [Real_X, Real_Y, Real_Z])
    return [Real_X, Real_Y, Real_Z]


def drawing(frame_queue, detections_queue, fps_queue, depth_queue):
	global drawing_queue
	global red_distance_queue
	global nlp_status
	random.seed(3)  # deterministic bbox colors
	video = set_saved_video(cap, args.out_filename, (width, height))
	global objectId
	while True:
		frame_resized = frame_queue.get()
		detections = detections_queue.get()
		fps = fps_queue.get()
		depth = depth_queue.get()  ###
		# depth np.array 타입
		if frame_resized is not None:
			class_colors.update(class_colors_mask)
			image = darknet.draw_boxes(detections, frame_resized, class_colors)
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

			id_check = []
			for j in range(len(red_distance_queue)):
				temp_centroid = red_distance_queue.pop()  # [(x1,y1,Id1), (x2,y2,Id2)]
				a1, b1, id1, w1, h1, a2, b2, id2, w2, h2 = \
					temp_centroid[0][0], temp_centroid[0][1], temp_centroid[0][2], temp_centroid[0][3], \
					temp_centroid[0][4], \
					temp_centroid[1][0], temp_centroid[1][1], temp_centroid[1][2], temp_centroid[1][3], \
					temp_centroid[1][4]
				cv2.line(image, (a1, b1), (a2, b2), (0, 0, 255), 2)
				xmin1 = int(round(a1 - (w1 / 2)))
				xmax1 = int(round(a1 + (w1 / 2)))
				ymin1 = int(round(b1 - (h1 / 2)))
				ymax1 = int(round(b1 + (h1 / 2)))

				xmin2 = int(round(a2 - (w2 / 2)))
				xmax2 = int(round(a2 + (w2 / 2)))
				ymin2 = int(round(b2 - (h2 / 2)))
				ymax2 = int(round(b2 + (h2 / 2)))

				cv2.rectangle(image, (xmin1, ymin1), (xmax1, ymax1), (0, 0, 255), 2)
				cv2.rectangle(image, (xmin2, ymin2), (xmax2, ymax2), (0, 0, 255), 2)
				if id1 not in id_check:
					id_check.append(id1)
				if id2 not in id_check:
					id_check.append(id2)

			drawing_queue.reverse()
			for i in range(len(drawing_queue)):
				temp = drawing_queue.pop()
				x, y, z, Real_ObjectId, h = temp[0], temp[1], temp[2], temp[3], temp[4]
				cv2.circle(image, (x, y), 3, (0, 255, 0), cv2.FILLED, cv2.LINE_4)
				cv2.putText(image, str(Real_ObjectId), (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

			distance_count_text = "Warning Distance: %s" % str(len(id_check))
			cv2.putText(image, distance_count_text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2,
						cv2.LINE_AA)

			images = np.hstack((image, depth))
			if args.out_filename is not None:
				video.write(image)
			if not args.dont_show:
				cv2.imshow('Inference', images)
			if cv2.waitKey(fps) == 27:
				break
	cap.release()
	cv2.destroyAllWindows()

def init_make():
    tts_in = gTTS(text='고객님 주변 사람과 일미터 간격을 유지해 주세요.', lang='ko')
    tts_in.save("./save_mp3_file/2m_input.mp3")
    os.system('ffmpeg -i ./save_mp3_file/2m_input.mp3 -acodec pcm_s16le -ac 1 -ar 16000 ./save_mp3_file/2m_output.wav')

def tts_speak(tts_queue):
    global speaking_state
    global tts_race
    speaking_state = True
    path = tts_queue
    chunk = 1024
    try:
        with wave.open(path, 'rb') as f:
            p = pyaudio.PyAudio()
            stream = p.open(format=p.get_format_from_width(f.getsampwidth()),
                            channels=f.getnchannels(),
                            rate=f.getframerate(),
                            output=True)
            data = f.readframes(chunk)
            print('start')
            while data:
                stream.write(data)
                data = f.readframes(chunk)
            stream.stop_stream()
            stream.close()
            p.terminate()
            # tts_race = False
            # speaking_state = False
    except Exception:
        print('error')
        # time.sleep(3)
        p.terminate()
        # speaking_state = False

def tts_speak2(tts_distance_queue):
    global speaking_state2
    global tts_distance
    speaking_state2 = True
    path = tts_distance_queue
    chunk = 1024
    try:
        with wave.open(path, 'rb') as f_dis:
            p_dis = pyaudio.PyAudio()
            stream_dis = p_dis.open(format=p_dis.get_format_from_width(f_dis.getsampwidth()),
                            channels=f_dis.getnchannels(),
                            rate=f_dis.getframerate(),
                            output=True)
            data = f_dis.readframes(chunk)
            print('start')
            while data:
                stream_dis.write(data)
                data = f_dis.readframes(chunk)
            stream_dis.stop_stream()
            stream_dis.close()
            p_dis.terminate()
            # del (tts_distance[:])
            # tts_distance = False
            # speaking_state2 = False

    except Exception:
        print('error')
        p_dis.terminate()
        # time.sleep(3)
        # speaking_state2 = False
        
pub = rospy.Publisher('mask_check', Int32, queue_size =10)
rospy.init_node('talker', anonymous = True)

def talker(bot_onoff):
    try:
        global robot_status_index
        if robot_status_index != 0:
            if not rospy.is_shutdown():
                rospy.loginfo(bot_onoff)
                pub.publish(bot_onoff)
        #rate = rospy.Rate(10)
        #rate.sleep() 
    except rospy.ROSInterruptException:
        pass
    except rospy.exceptions.ROSException as e:
        print("Node has already been initialized, do nothing")

def callback2(data):
	global robot_status_index
	rospy.loginfo("robot status: %d", data.data)
	if data.data == 0:
		robot_status_index = 0
	else:
		robot_status_index = 1
		
def callback3(data):
	global nlp_status
	rospy.loginfo("nlp status: %d", data.data)
	if data.data == 3:
		nlp_status = 0
	elif data.data == 4:
		nlp_status = 1
			
def listener():
    rospy.Subscriber("robot_status", Int32, callback2)
    rospy.Subscriber("nlp_status", Int32, callback3)
    #rospy.spin()	


def visions():
    frame_queue = Queue()
    darknet_image_queue = Queue(maxsize=1)
    detections_queue = Queue(maxsize=1)
    fps_queue = Queue(maxsize=1)
    depth_queue = Queue(maxsize=1)
    distance_queue = Queue(maxsize=1)
    tts_queue = Queue(maxsize=1)
    tts_distance_queue  = Queue(maxsize=1)
    Thread(target=video_capture, args=(frame_queue, darknet_image_queue, depth_queue, distance_queue)).start()
    Thread(target=inference, args=(darknet_image_queue, detections_queue, fps_queue, distance_queue)).start()
    Thread(target=drawing, args=(frame_queue, detections_queue, fps_queue, depth_queue)).start()


input_data =1
drawing_queue = []
tts_race = False
tts_distance = False
speaking_state = False
speaking_state2 = False
red_distance_queue = []
#init_make()
args = parser()
# args를 parser로 받아와서 argument error를 건다.

check_arguments_errors(args)
network, class_names, class_colors = darknet.load_network(
	args.config_file,
	args.data_file,
	args.weights,
	batch_size=1
)

network_mask, class_names_mask, class_colors_mask = darknet.load_network(
	"/media/hustar/Hustar/darknet/mask_set/yolov4-obj.cfg",
	"/media/hustar/Hustar/darknet/mask_set/obj.data",
	"/media/hustar/Hustar/darknet/mask_set/yolov4-obj_best.weights",
	batch_size=1
)

# Darknet doesn't accept numpy images.
# Create one with image we reuse for each detect
width = darknet.network_width(network)
height = darknet.network_height(network)
darknet_image = darknet.make_image(width, height, 3)
align_to = rs.stream.color
align = rs.align(align_to)
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

# Start streaming
pipeline.start(config)
cap = pipeline.wait_for_frames()

def main():
    #listener()
    visions()
    
if __name__ == '__main__':
    main()
