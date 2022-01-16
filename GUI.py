import sys
import os
import cv2
import time
import numpy as np
import random
from PyQt5.QtWidgets import QMainWindow,QLabel,QApplication,QPushButton,QFileDialog,QProgressBar,QLineEdit,QMessageBox,QWidget,QGraphicsOpacityEffect,QComboBox
from PyQt5.QtGui import QPixmap,QFont,QMovie,QPainter,QBrush,QPen
from PyQt5.QtCore import Qt, QThread,pyqtSignal,QByteArray
import pickle

def image_pose_estimation(IMAGE_PATH,MODE,DEVICE):
	left_arm = []
	left_fore_arm = []
	left_wrist = []

	right_arm = []
	right_fore_arm = []
	right_wrist = []

	left_up_leg = []
	left_leg = []
	left_ankle = []

	right_up_leg = []
	right_leg = []
	right_ankle = []
	protoFile =  ""
	weightsFile = ""
	
	if MODE == "COCO":
		protoFile = "models/pose/coco/pose_deploy_linevec.prototxt"
		weightsFile = "models/pose/coco/pose_iter_440000.caffemodel"
		nPoints = 18
		POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

		left_arm = 5
		left_fore_arm = 6
		left_wrist = 7

		right_arm = 2
		right_fore_arm = 3
		right_wrist = 4

		left_up_leg = 11
		left_leg = 12
		left_ankle = 13

		right_up_leg = 8
		right_leg = 9
		right_ankle = 10

	elif MODE == "MPI" :
		protoFile = "models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
		weightsFile = "models/pose/mpi/pose_iter_160000.caffemodel"
		nPoints = 15
		POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]

		left_arm = 5
		left_fore_arm = 6
		left_wrist = 7

		right_arm = 2
		right_fore_arm = 3
		right_wrist = 4

		left_up_leg = 11
		left_leg = 12
		left_ankle = 13

		right_up_leg = 8
		right_leg = 9
		right_ankle = 10

	elif MODE == "BODY_25" :
		protoFile = "models/pose/body_25/pose_deploy.prototxt"
		weightsFile = "models/pose/body_25/pose_iter_584000.caffemodel"
		nPoints = 25
		POSE_PAIRS = [[0,16],[16,18],[0,15],[15,17],[0,1],[1,2],[2,3],[3,4],[1,5],[5,6],[6,7],[1,8],[8,9],[9,10],[10,11],[8,12],[12,13],[13,14],[11,24],[11,22],[14,21],[14,19],[22,23],[19,20] ]

		left_arm = 5
		left_fore_arm = 6
		left_wrist = 7

		right_arm = 2
		right_fore_arm = 3
		right_wrist = 4

		left_up_leg = 12
		left_leg = 13
		left_ankle = 14

		right_up_leg = 9
		right_leg = 10
		right_ankle = 11


	frame = cv2.imread(IMAGE_PATH)
	frameCopy = np.copy(frame)
	frameWidth = frame.shape[1]
	frameHeight = frame.shape[0]
	threshold = 0.1

	net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

	if DEVICE == "cpu":
		net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
		print("Using CPU device")
	elif DEVICE == "gpu":
		net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
		net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
		print("Using GPU device")

	t = time.time()
	# input image dimensions for the network
	inWidth = 368
	inHeight = 368
	inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
											(0, 0, 0), swapRB=False, crop=False)

	net.setInput(inpBlob)

	output = net.forward()
	print("time taken by network : {:.3f}".format(time.time() - t))

	H = output.shape[2]
	W = output.shape[3]

	# Empty list to store the detected keypoints
	points = []

	for i in range(nPoints):
		# confidence map of corresponding body's part.
		probMap = output[0, i, :, :]

		# Find global maxima of the probMap.
		minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        
		# Scale the point to fit on the original image
		x = (frameWidth * point[0]) / W
		y = (frameHeight * point[1]) / H

		if prob > threshold : 
			cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
			cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 250), 2, lineType=cv2.LINE_AA)

			# Add the point to the list if the probability is greater than the threshold
			points.append((int(x), int(y)))
		else :
			points.append(None)
		print(i,"["+(str(int(x))+","+str(int(y))+"]"))

		if i == left_arm:
			container["LEFT_ARM"] = list((int(x), int(y)))
		elif i == left_fore_arm:
			container["LEFT_FORE_ARM"] = list((int(x), int(y)))
		elif i == left_wrist:
			container["LEFT_WRIST"] = list((int(x), int(y)))
		elif i == right_arm:
			container["RIGHT_ARM"] = list((int(x), int(y)))
		elif i == right_fore_arm:
			container["RIGHT_FORE_ARM"] = list((int(x), int(y)))
		elif i == right_wrist:
			container["RIGHT_WRIST"] = list((int(x), int(y)))
		elif i == left_up_leg:
			container["LEFT_UP_LEG"] = list((int(x), int(y)))
		elif i == left_leg:
			container["LEFT_LEG"] = list((int(x), int(y)))
		elif i == left_ankle:
			container["LEFT_ANKLE"] = list((int(x), int(y)))
		elif i == right_up_leg:
			container["RIGHT_UP_LEG"] = list((int(x), int(y)))
		elif i == right_leg:
			container["RIGHT_LEG"] = list((int(x), int(y)))
		elif i == right_ankle:
			container["RIGHT_ANKLE"] = list((int(x), int(y)))
	container_list.append(container)

	# Draw Skeleton
	for pair in POSE_PAIRS:
		partA = pair[0]
		partB = pair[1]

		if points[partA] and points[partB]:
			cv2.line(frame, points[partA], points[partB], (0,220,0), 5)
			cv2.circle(frame, points[partA], 8, (0,0,255), thickness=-1, lineType=cv2.FILLED)
			cv2.circle(frame, points[partB], 8, (255,0,0), thickness=-1, lineType=cv2.FILLED)


	cv2.imwrite('Output-Keypoints.jpg', frameCopy)
	cv2.imwrite('Output-Skeleton.jpg', frame)

	print("Total time taken : {:.3f}".format(time.time() - t))

	cv2.waitKey(0)

def video_pose_estimation(VIDEO_PATH,MODE,DEVICE):
	left_arm = []
	left_fore_arm = []
	left_wrist = []

	right_arm = []
	right_fore_arm = []
	right_wrist = []

	left_up_leg = []
	left_leg = []
	left_ankle = []

	right_up_leg = []
	right_leg = []
	right_ankle = []

	if MODE == "COCO":
		protoFile = "models/pose/coco/pose_deploy_linevec.prototxt"
		weightsFile = "models/pose/coco/pose_iter_440000.caffemodel"
		nPoints = 18
		POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

		left_arm = 5
		left_fore_arm = 6
		left_wrist = 7

		right_arm = 2
		right_fore_arm = 3
		right_wrist = 4

		left_up_leg = 11
		left_leg = 12
		left_ankle = 13

		right_up_leg = 8
		right_leg = 9
		right_ankle = 10

	elif MODE == "MPI" :
		protoFile = "models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
		weightsFile = "models/pose/mpi/pose_iter_160000.caffemodel"
		nPoints = 15
		POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]

		left_arm = 5
		left_fore_arm = 6
		left_wrist = 7

		right_arm = 2
		right_fore_arm = 3
		right_wrist = 4

		left_up_leg = 11
		left_leg = 12
		left_ankle = 13

		right_up_leg = 8
		right_leg = 9
		right_ankle = 10


	inWidth = 368
	inHeight = 368
	threshold = 0.1


	input_source = VIDEO_PATH
	cap = cv2.VideoCapture(input_source)
	hasFrame, frame = cap.read()

	vid_writer = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame.shape[1],frame.shape[0]))

	net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
	if DEVICE == "cpu":
		net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
		print("Using CPU device")
	elif DEVICE == "gpu":
		net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
		net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
		print("Using GPU device")

	while cv2.waitKey(1) < 0:
		t = time.time()
		hasFrame, frame = cap.read()
		frameCopy = np.copy(frame)
		if not hasFrame:
			cv2.waitKey()
			break

		frameWidth = frame.shape[1]
		frameHeight = frame.shape[0]

		inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
								  (0, 0, 0), swapRB=False, crop=False)
		net.setInput(inpBlob)
		output = net.forward()

		H = output.shape[2]
		W = output.shape[3]
		# Empty list to store the detected keypoints
		points = []

		for i in range(nPoints):
			# confidence map of corresponding body's part.
			probMap = output[0, i, :, :]

			# Find global maxima of the probMap.
			minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
			
			# Scale the point to fit on the original image
			x = (frameWidth * point[0]) / W
			y = (frameHeight * point[1]) / H

			if prob > threshold : 
				cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
				cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

				# Add the point to the list if the probability is greater than the threshold
				points.append((int(x), int(y)))
			else :
				points.append(None)

			if i == left_arm:
				container["LEFT_ARM"] = list((int(x), int(y)))
				print(list((int(x), int(y))))
			elif i == left_fore_arm:
				container["LEFT_FORE_ARM"] = list((int(x), int(y)))
			elif i == left_wrist:
				container["LEFT_WRIST"] = list((int(x), int(y)))
			elif i == right_arm:
				container["RIGHT_ARM"] = list((int(x), int(y)))
			elif i == right_fore_arm:
				container["RIGHT_FORE_ARM"] = list((int(x), int(y)))
			elif i == right_wrist:
				container["RIGHT_WRIST"] = list((int(x), int(y)))
			elif i == left_up_leg:
				container["LEFT_UP_LEG"] = list((int(x), int(y)))
			elif i == left_leg:
				container["LEFT_LEG"] = list((int(x), int(y)))
			elif i == left_ankle:
				container["LEFT_ANKLE"] = list((int(x), int(y)))
			elif i == right_up_leg:
				container["RIGHT_UP_LEG"] = list((int(x), int(y)))
			elif i == right_leg:
				container["RIGHT_LEG"] = list((int(x), int(y)))
			elif i == right_ankle:
				container["RIGHT_ANKLE"] = list((int(x), int(y)))
		container_list.append(container)

		# Draw Skeleton
		for pair in POSE_PAIRS:
			partA = pair[0]
			partB = pair[1]

			if points[partA] and points[partB]:
				cv2.line(frame, points[partA], points[partB], (0, 255, 255), 3, lineType=cv2.LINE_AA)
				cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
				cv2.circle(frame, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

		cv2.putText(frame, "time taken = {:.2f} sec".format(time.time() - t), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
		# cv2.putText(frame, "OpenPose using OpenCV", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 50, 0), 2, lineType=cv2.LINE_AA)
		# cv2.imshow('Output-Keypoints', frameCopy)
		cv2.imshow('Output-Skeleton', frame)

		vid_writer.write(frame)
	vid_writer.release()

X = 400
Y = 150
WIDTH = 1200
HEIGHT = 800

container_list = []
container = {"LEFT_ARM" : [],
			"LEFT_FORE_ARM" : [],
			"LEFT_WRIST" : [],

			"RIGHT_ARM" : [],
			"RIGHT_FORE_ARM" : [],
			"RIGHT_WRIST" : [],

			"LEFT_UP_LEG" : [],
			"LEFT_LEG" : [],
			"LEFT_ANKLE" : [],

			"RIGHT_UP_LEG" : [],
			"RIGHT_LEG" : [],
			"RIGHT_ANKLE" : []}

class Template():
	def on_welcome_btn_click(self):
		controller.current_screen = "home"
		self.parent.switch_to_home.emit()
		self.parent.close()

	def on_choose_input_btn_click(self):
		controller.current_screen = "input"
		self.parent.switch_to_input.emit()
		self.parent.close()

	def on_configure_network_btn_click(self):
		controller.current_screen = "network"
		self.parent.switch_to_network.emit()
		self.parent.close()

	def on_motion_capture_btn_click(self):
		controller.current_screen = "capture"
		self.parent.switch_to_capture.emit()
		self.parent.close()

	def on_transfer_data_btn_click(self):
		controller.current_screen = "transfer"
		self.parent.switch_to_transfer.emit()
		self.parent.close()

	def __init__(self,parent):
		super().__init__()
		self.parent = parent
		self.createUI()

	def update_active_screen(self):
		self.format_side_bar()
		button = self.welcome_btn
		if controller.current_screen == "home":
			button = self.welcome_btn
		elif controller.current_screen == "input":
			button = self.choose_input_btn
		elif controller.current_screen == "network":
			button = self.configure_network_btn
		elif controller.current_screen == "capture":
			button = self.motion_capture_btn
		elif controller.current_screen == "transfer":
			button = self.transfer_data_btn
		button.setStyleSheet("QPushButton"
                             "{"
                             "background-color : #F77142;color:#FFFEF6;font-weight:bold;border:3px solid white;font-size:8.7pt;border-radius : 15px;"
                             "}")

	def format_side_bar(self):
		start_x = 35
		start_y = 140
		button_width = 180
		button_height = 67
		for button in self.button_list:
			button.resize(button_width,button_height)
			button.move(start_x,start_y)
			button.setStyleSheet("QPushButton::hover"
                             "{"
                             "background-color : #00617F;color:#FFFEF6;font-weight:bold;border:2px solid white"
                             "}"
                             "QPushButton"
                             "{"
                             "font-size:8.7pt;border-radius : 15px;background-color:#104070;color:#FFFEF6;font-weight:bold"
                             "}"
                             )
			start_y+=100

	def createUI(self):
		self.parent.setWindowTitle("Motion Capture With Computer Vision")
		self.footer_opacity_effect = QGraphicsOpacityEffect()
		self.footer_opacity_effect.setOpacity(0.8)
		self.panel_opacity_effect = QGraphicsOpacityEffect()
		self.panel_opacity_effect.setOpacity(0.8)

		self.background=QLabel(self.parent)
		self.background_pixmap=QPixmap('abstract.jpg')
		self.background_pixmap = self.background_pixmap.scaledToWidth(WIDTH)
		self.background.move(0,0)
		self.background.setPixmap(self.background_pixmap)
		
		self.buttons_panel = QLabel(self.parent)
		self.buttons_panel.move(0,0)
		self.buttons_panel.resize(270,HEIGHT-50)
		self.buttons_panel.setStyleSheet("background-color : #1A7734")
		self.buttons_panel.setGraphicsEffect(self.panel_opacity_effect)
		self.logo=QLabel(self.parent)
		self.logo_pixmap=QPixmap('logo_large.png')
		self.logo_pixmap = self.logo_pixmap.scaledToWidth(250)
		self.logo.move(5,8)
		self.logo.setPixmap(self.logo_pixmap)
		self.logo.setStyleSheet("border:5px solid white;border-radius:15px")

		self.welcome_btn = QPushButton(self.parent)
		self.welcome_btn.setText("Get Started")
		self.welcome_btn.clicked.connect(self.on_welcome_btn_click)

		self.choose_input_btn = QPushButton(self.parent)
		self.choose_input_btn.setText("Choose Input Source")
		self.choose_input_btn.clicked.connect(self.on_choose_input_btn_click)

		self.configure_network_btn = QPushButton(self.parent)
		self.configure_network_btn.setText("Neural Network")
		self.configure_network_btn.clicked.connect(self.on_configure_network_btn_click)

		self.motion_capture_btn = QPushButton(self.parent)
		self.motion_capture_btn.setText("Capture Motion Data")
		self.motion_capture_btn.clicked.connect(self.on_motion_capture_btn_click)

		self.transfer_data_btn = QPushButton(self.parent)
		self.transfer_data_btn.setText("Choose 3D Model")
		self.transfer_data_btn.clicked.connect(self.on_transfer_data_btn_click)

		self.footer_panel = QLabel(self.parent)
		self.footer_panel.move(0,HEIGHT-50)
		self.footer_panel.resize(WIDTH,50)
		self.footer_panel.setStyleSheet("background-color : rgb(20,20,20)")
		self.footer_panel.setGraphicsEffect(self.footer_opacity_effect)
		self.python=QLabel(self.parent)
		self.python_pixmap=QPixmap('Python-Logo-Png.png')
		#python_pixmap = python_pixmap.scaledToHeight(35)
		self.python.setPixmap(self.python_pixmap)
		self.python.resize(230,60)
		self.python.move(640,HEIGHT-55)

		self.button_list = [self.welcome_btn,self.choose_input_btn,self.configure_network_btn,self.motion_capture_btn,self.transfer_data_btn]
		self.update_active_screen()
		


class HomeScreen(QWidget):
	switch_to_home = pyqtSignal()
	switch_to_input = pyqtSignal()
	switch_to_network = pyqtSignal()
	switch_to_capture = pyqtSignal()
	switch_to_transfer = pyqtSignal()

	def __init__(self):
		super().__init__()
		self.createUI()

	def createUI(self):
		self.setGeometry(X,Y,WIDTH,HEIGHT)
		self.template = Template(self)
		self.character_opacity_effect = QGraphicsOpacityEffect()
		self.character_opacity_effect.setOpacity(0.8)
		self.opacity_effect = QGraphicsOpacityEffect()
		self.opacity_effect.setOpacity(0.8)
		self.ui_panel = QLabel(self)
		self.ui_panel.move(315,150)
		self.ui_panel.resize(840,550)
		self.ui_panel.setStyleSheet("background-color : rgb(20,20,20);border-radius:20px")
		self.ui_panel.setGraphicsEffect(self.opacity_effect)

		self.brand=QLabel(self)
		self.brand_pixmap=QPixmap('title.png')
		self.brand_pixmap = self.brand_pixmap.scaledToWidth(270)
		self.brand.move(580,25)
		self.brand.setPixmap(self.brand_pixmap)
		self.brand.setStyleSheet("border-radius:5px;border:3px solid darkgray")
		
		logo_offset = 70
		self.logo=QLabel(self)
		self.logo_pixmap=QPixmap('gif_frame.png')
		self.logo.move(490,100+logo_offset+15)
		self.logo.setPixmap(self.logo_pixmap)
		self.character=QLabel(self)
		self.character.resize(300,300)
		self.movie=QMovie("gif_character.gif",QByteArray(),self)
		self.movie.setSpeed(100)
		self.movie.start()
		self.character.setMovie(self.movie)
		self.character.move(600,190+logo_offset-15)
		self.welcome = QLabel(self)
		self.welcome.setText("Welcome!")
		self.welcome.setStyleSheet("font-family : Bahnschrift SemiBold; color : rgb(255,255,255); font-size : 35pt")
		self.welcome.resize(250,40)
		self.welcome.move(630,410+logo_offset)
		self.message = QLabel(self)
		self.message.setText("-- Let's get started --")
		self.message.setStyleSheet("font-family : Bahnschrift SemiBold; color : rgb(255,255,255); font-size : 16pt")
		self.message.resize(300,30)
		self.message.move(630,470+logo_offset)

class InputScreen(QWidget):
	switch_to_home = pyqtSignal()
	switch_to_input = pyqtSignal()
	switch_to_network = pyqtSignal()
	switch_to_capture = pyqtSignal()
	switch_to_transfer = pyqtSignal()

	def __init__(self):
		super().__init__()
		self.createUI()

	def format_buttons(self):
		start_x = 460
		start_y = 400
		button_width = 140
		button_height = 50
		for button in self.button_list:
			button.resize(button_width,button_height)
			button.move(start_x,start_y)
			button.setStyleSheet("QPushButton::hover"
                             "{"
                             "background-color : #F77142;color:#FFFEF6;font-weight:bold;border:2px solid white"
                             "}"
                             "QPushButton"
                             "{"
                             "font-size:9pt;border-radius : 15px;background-color : rgb(20,80,180);font-weight:bold;color:white"
                             "}"
                             )
			start_x+=170

	def openImageFileDialog(self):
		options = QFileDialog.Options()
		fileName, _ = QFileDialog.getOpenFileName(self,"Select File", "","Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)", options=options)
		if fileName:
			controller.image_path = fileName
			controller.input_source = "image"
			self.path_lbl.setText(fileName)
			container_list.clear()

	def openVideoFileDialog(self):
		options = QFileDialog.Options()
		fileName, _ = QFileDialog.getOpenFileName(self,"Select File", "","Video Files (*.mp4 *.webp);;All Files (*)", options=options)
		if fileName:
			controller.video_path = fileName
			controller.input_source = "video"
			self.path_lbl.setText(fileName)
			container_list.clear()

	def on_start_webcam_btn_click(self):
		os.system("start microsoft.windows.camera:")

	def on_select_image_btn_click(self):
		self.openImageFileDialog()

	def on_select_video_btn_click(self):
		self.openVideoFileDialog()

	def createUI(self):
		self.setGeometry(X,Y,WIDTH,HEIGHT)
		self.template = Template(self)
		self.opacity_effect = QGraphicsOpacityEffect()
		self.opacity_effect.setOpacity(0.8)
		self.ui_panel = QLabel(self)
		self.ui_panel.move(350,200)
		self.ui_panel.resize(750,350)
		self.ui_panel.setStyleSheet("background-color : rgb(20,20,20);border-radius:20px")
		self.ui_panel.setGraphicsEffect(self.opacity_effect)
		self.title_lbl = QLabel(self.ui_panel)
		self.title_lbl.setText("Select Input Source")
		self.title_lbl.setStyleSheet("color:white;font-size:30pt;margin-left:210px;margin-top:20px;font-family:Bahnschrift SemiBold")
		self.brand=QLabel(self)
		self.brand_pixmap=QPixmap('title.png')
		self.brand_pixmap = self.brand_pixmap.scaledToWidth(270)
		self.brand.move(580,50)
		self.brand.setPixmap(self.brand_pixmap)
		self.brand.setStyleSheet("border-radius:5px;border:3px solid darkgray")
		self.path_lbl = QLabel(self.ui_panel)
		if controller.input_source == "image":
			self.path_lbl.setText(controller.image_path)
		elif controller.input_source == "video":
			self.path_lbl.setText(controller.video_path)
		self.path_lbl.resize(700,152)
		self.path_lbl.setStyleSheet("color:black;font-size:12pt;border:1px solid white;padding:10px;margin-left:50px;margin-top:100px;background-color:white")
		
		self.select_image_btn = QPushButton(self)
		self.select_image_btn.setText("Select Image")
		self.select_image_btn.clicked.connect(self.on_select_image_btn_click)

		self.select_video_btn = QPushButton(self)
		self.select_video_btn.setText("Select Video")
		self.select_video_btn.clicked.connect(self.on_select_video_btn_click)

		self.start_webcam_btn = QPushButton(self)
		self.start_webcam_btn.setText("Start Webcam")
		self.start_webcam_btn.clicked.connect(self.on_start_webcam_btn_click)

		self.button_list = [self.select_image_btn,self.select_video_btn,self.start_webcam_btn]
		self.format_buttons()

class NetworkScreen(QWidget):
	switch_to_home = pyqtSignal()
	switch_to_input = pyqtSignal()
	switch_to_network = pyqtSignal()
	switch_to_capture = pyqtSignal()
	switch_to_transfer = pyqtSignal()

	def __init__(self):
		super().__init__()
		self.createUI()

	def update_keypoints(self):
		if self.model_combobox.currentText() == "MPI":
			self.keypoints_lbl.setText("Body keypoints      "+str(15))
		elif self.model_combobox.currentText() == "COCO":
			self.keypoints_lbl.setText("Body keypoints      "+str(18))
		elif self.model_combobox.currentText() == "BODY_25":
			self.keypoints_lbl.setText("Body keypoints      "+str(25))

	def update_model(self):
		self.update_keypoints()
		controller.model = self.model_combobox.currentText()

	def update_pose(self):
		controller.pose = self.pose_combobox.currentText()

	def update_device(self):
		controller.device = self.device_combobox.currentText()

	def update_camera(self):
		controller.camera = self.camera_direction_combobox.currentText()

	def createUI(self):
		self.setGeometry(X,Y,WIDTH,HEIGHT)
		self.template = Template(self)
		self.opacity_effect = QGraphicsOpacityEffect()
		self.opacity_effect.setOpacity(0.8)
		self.ui_panel = QLabel(self)
		self.ui_panel.move(350,200)
		self.ui_panel.resize(750,400)
		self.ui_panel.setStyleSheet("background-color : rgb(20,20,20);border-radius:20px")
		self.ui_panel.setGraphicsEffect(self.opacity_effect)
		self.title_lbl = QLabel(self.ui_panel)
		self.title_lbl.setText("Configure Neural Network")
		self.title_lbl.setStyleSheet("color:white;font-size:30pt;margin-left:130px;margin-top:20px;font-family:Bahnschrift SemiBold")

		self.brand=QLabel(self)
		self.brand_pixmap=QPixmap('title.png')
		self.brand_pixmap = self.brand_pixmap.scaledToWidth(270)
		self.brand.move(580,50)
		self.brand.setPixmap(self.brand_pixmap)
		self.brand.setStyleSheet("border-radius:5px;border:3px solid darkgray")

		self.pose_lbl =QLabel(self.ui_panel)
		self.pose_lbl.setText("Pose Estimation")
		self.pose_lbl.setStyleSheet("color:white;font-size:12pt;margin-left:10px;margin-top:120px")
		self.pose_combobox = QComboBox(self)
		self.pose_combobox.addItems(["FullBody","Hand"])
		self.pose_combobox.move(500,317)
		self.pose_combobox.setStyleSheet("font-size:10pt")
		self.pose_combobox.currentIndexChanged.connect(self.update_pose)

		self.choose_model_lbl =QLabel(self.ui_panel)
		self.choose_model_lbl.setText("Network model")
		self.choose_model_lbl.setStyleSheet("color:white;font-size:12pt;margin-left:10px;margin-top:160px")
		self.model_combobox = QComboBox(self)
		self.model_combobox.addItems(["COCO","MPI","BODY_25"])
		self.model_combobox.move(500,357)
		self.model_combobox.setStyleSheet("font-size:10pt")
		self.model_combobox.currentIndexChanged.connect(self.update_model)
		
		self.keypoints_lbl =QLabel(self.ui_panel)
		self.keypoints_lbl.setStyleSheet("color:white;font-size:12pt;margin-left:10px;margin-top:200px")
		self.update_keypoints()

		self.choose_device_lbl =QLabel(self.ui_panel)
		self.choose_device_lbl.setText("Choose device")
		self.choose_device_lbl.setStyleSheet("color:white;font-size:12pt;margin-left:10px;margin-top:240px")
		self.device_combobox = QComboBox(self)
		self.device_combobox.addItems(["CPU","GPU"])
		self.device_combobox.move(500,437)
		self.device_combobox.setStyleSheet("font-size:10pt")
		self.device_combobox.currentIndexChanged.connect(self.update_device)

		self.camera_direction_lbl =QLabel(self.ui_panel)
		self.camera_direction_lbl.setText("Camera direction")
		self.camera_direction_lbl.setStyleSheet("color:white;font-size:12pt;margin-left:10px;margin-top:280px")
		self.camera_direction_combobox = QComboBox(self)
		self.camera_direction_combobox.addItems(["Frontal","Sideways"])
		self.camera_direction_combobox.move(500,477)
		self.camera_direction_combobox.setStyleSheet("font-size:10pt")
		self.camera_direction_combobox.currentIndexChanged.connect(self.update_camera)

class CaptureScreen(QWidget):
	switch_to_home = pyqtSignal()
	switch_to_input = pyqtSignal()
	switch_to_network = pyqtSignal()
	switch_to_capture = pyqtSignal()
	switch_to_transfer = pyqtSignal()

	def __init__(self):
		super().__init__()
		self.createUI()

	def format_buttons(self):
		start_x = 580
		start_y = 615
		button_width = 120
		button_height = 50
		for button in self.button_list:
			button.resize(button_width,button_height)
			button.move(start_x,start_y)
			button.setStyleSheet("QPushButton::hover"
                             "{"
                             "background-color :#FF5923 ;color:white;font-weight:bold;border:2px solid darkgray"
                             "}"
                             "QPushButton"
                             "{"
                             "font-size:9pt;border-radius : 15px;background-color : rgb(20,80,180);font-weight:bold;color:white"
                             "}")
			start_x+=150

	def set_output_image(self):
		if controller.scanned:
			try:
				self.output_pixmap=QPixmap("Output-Skeleton.JPG")
				if self.dimensions[0] >= self.dimensions[1]:
					self.output_pixmap = self.output_pixmap.scaledToHeight(380)
				else:
					self.output_pixmap = self.output_pixmap.scaledToWidth(380)
				self.output_image_lbl.setPixmap(self.output_pixmap)
			except:
				pass

	def image_pose_estimation(self):
		image_pose_estimation(controller.image_path,controller.model,controller.device)
		controller.scanned = True
		self.set_output_image()
		message_box = QMessageBox(self)
		message_box.setIcon(QMessageBox.Information)
		message_box.setText("Pose Estimation completed successfully")
		message_box.setWindowTitle("Success")
		message_box.setStandardButtons(QMessageBox.Ok)
		message_box.show()

	def video_pose_estimation(self):
		video_pose_estimation(controller.video_path,controller.model,controller.device)
		controller.scanned = True
		self.set_output_image()
		message_box = QMessageBox(self)
		message_box.setIcon(QMessageBox.Information)
		message_box.setText("Pose Estimation completed successfully")
		message_box.setWindowTitle("Success")
		message_box.setStandardButtons(QMessageBox.Ok)
		message_box.show()	

	def on_start_scan_btn_click(self):
		if controller.input_source == "image":
			self.image_pose_estimation()
		elif controller.input_source == "video":
			self.video_pose_estimation()

	def on_save_data_btn_click(self):
		with open('motion.pickle','wb') as file:
			pickle.dump(container_list,file)
			message_box = QMessageBox(self)
			message_box.setIcon(QMessageBox.Information)
			message_box.setText("Captured data saved successfully")
			message_box.setWindowTitle("Save successfull")
			message_box.setStandardButtons(QMessageBox.Ok)
			message_box.show()

	def createUI(self):
		try:
			img = cv2.imread(controller.image_path)
			self.dimensions = img.shape
		except:
			pass
		self.setGeometry(X,Y,WIDTH,HEIGHT)
		self.template = Template(self)
		self.opacity_effect = QGraphicsOpacityEffect()
		self.opacity_effect.setOpacity(0.8)
		self.ui_panel = QLabel(self)
		self.ui_panel.move(315,80)
		self.ui_panel.resize(840,600)
		self.ui_panel.setStyleSheet("background-color : rgb(20,20,20);border-radius:20px")
		self.ui_panel.setGraphicsEffect(self.opacity_effect)
		self.title_lbl = QLabel(self.ui_panel)
		self.title_lbl.setText("Capture Motion Data")
		self.title_lbl.setStyleSheet("color:white;font-size:30pt;margin-left:200px;margin-top:20px;font-family:Bahnschrift SemiBold")
		self.input_lbl = QLabel(self.ui_panel)
		self.input_lbl.setText("Input Image")
		self.input_lbl.setStyleSheet("color:white;font-size:18pt;margin-left:150px;margin-top:80px;font-family:Bahnschrift SemiBold")
		self.output_lbl = QLabel(self.ui_panel)
		self.output_lbl.setText("Output Image")
		self.output_lbl.setStyleSheet("color:white;font-size:18pt;margin-left:550px;margin-top:80px;font-family:Bahnschrift SemiBold")


		self.input_image_lbl = QLabel(self)
		self.input_image_lbl.resize(400,400)
		self.input_image_lbl.move(330,200)
		self.input_image_lbl.setStyleSheet("background-color:black;border-radius:10px")
		self.input_image_lbl.setAlignment(Qt.AlignCenter)
		try:
			self.input_pixmap=QPixmap(controller.image_path)
			if self.dimensions[0] >= self.dimensions[1]:
				self.input_pixmap = self.input_pixmap.scaledToHeight(380)
			else:
				self.input_pixmap = self.input_pixmap.scaledToWidth(380)
			self.input_image_lbl.setPixmap(self.input_pixmap)
		except:
			pass
		self.output_image_lbl = QLabel(self)
		self.output_image_lbl.resize(400,400)
		self.output_image_lbl.move(740,200)
		self.output_image_lbl.setStyleSheet("background-color:black;border-radius:10px")
		self.output_image_lbl.setAlignment(Qt.AlignCenter)
		self.set_output_image()

		self.start_scan_btn = QPushButton(self)
		self.start_scan_btn.setText("Start Scan")
		self.start_scan_btn.clicked.connect(self.on_start_scan_btn_click)
		self.save_data_btn = QPushButton(self)
		self.save_data_btn.setText("Save Data")
		self.save_data_btn.clicked.connect(self.on_save_data_btn_click)

		self.button_list = [self.start_scan_btn,self.save_data_btn]
		self.format_buttons()

class TransferScreen(QWidget):
	switch_to_home = pyqtSignal()
	switch_to_input = pyqtSignal()
	switch_to_network = pyqtSignal()
	switch_to_capture = pyqtSignal()
	switch_to_transfer = pyqtSignal()

	def __init__(self):
		super().__init__()
		self.createUI()

	def on_choose_model_btn_click(self):
		self.openModelFileDialog()

	def on_transfer_data_btn_click(self):
		text = "blender "+controller.character_path
		print(text)
		text = text.replace("\\","/")
		os.system(text)

	def openModelFileDialog(self):
		options = QFileDialog.Options()
		fileName, _ = QFileDialog.getOpenFileName(self,"Select File", "","3D Model Files (*.blend);;All Files (*)", options=options)
		if fileName:
			controller.character_path = fileName

	def format_buttons(self):
		start_x = 850
		start_y = 645
		button_width = 115
		button_height = 45
		for button in self.button_list:
			button.resize(button_width,button_height)
			button.move(start_x,start_y)
			button.setStyleSheet("QPushButton::hover"
                             "{"
                             "background-color :#FF5923 ;color:white;font-weight:bold;border:2px solid darkgray"
                             "}"
                             "QPushButton"
                             "{"
                             "font-size:8pt;border-radius : 12px;background-color : rgb(20,80,180);font-weight:bold;color:white"
                             "}")
			start_x+=130

	def format_labels(self):
		start_x = 820
		start_y = 190
		for label in self.label_list:
			label.setStyleSheet("color:white;font-size:12pt;padding-top:3px;")
			label.move(start_x,start_y)
			start_y+=33

	def format_edit(self):
		start_x = 940
		start_y = 190
		for edit in self.edit_list:
			edit.move(start_x,start_y)
			edit.setStyleSheet("font-size:11pt;font-family:calibri;")
			edit.resize(170,25)
			start_y+=33

	def createUI(self):
		img = cv2.imread("master.png")
		dimensions = img.shape
		self.setGeometry(X,Y,WIDTH,HEIGHT)
		self.template = Template(self)
		self.opacity_effect = QGraphicsOpacityEffect()
		self.opacity_effect.setOpacity(0.8)
		self.ui_panel = QLabel(self)
		self.ui_panel.move(315,40)
		self.ui_panel.resize(840,670)
		self.ui_panel.setStyleSheet("background-color : rgb(20,20,20);border-radius:20px")
		self.ui_panel.setGraphicsEffect(self.opacity_effect)
		self.title_lbl = QLabel(self.ui_panel)
		self.title_lbl.setText("Transfer Motion Data")
		self.title_lbl.setStyleSheet("color:white;font-size:30pt;margin-left:200px;margin-top:20px;font-family:Bahnschrift SemiBold")
		self.controls_panel = QLabel(self)
		self.controls_panel.resize(333,500)
		self.controls_panel.move(800,130)
		self.controls_panel.setStyleSheet("background-color:rgb(10,10,10)")
		self.model_image_lbl = QLabel(self)
		self.model_image_lbl.resize(450,560)
		self.model_image_lbl.move(330,130)
		self.model_image_lbl.setStyleSheet("background-color:rgb(10,10,10);padding-left:10px;padding-top:5px;border-radius:5px")
		self.model_pixmap=QPixmap('master.png')
		self.model_pixmap = self.model_pixmap.scaledToHeight(540)
		self.model_image_lbl.setPixmap(self.model_pixmap)
		
		self.instruction_lbl = QLabel(self)
		self.instruction_lbl.setText("Configure Bone Names")
		self.instruction_lbl.move(810,140)
		self.instruction_lbl.setStyleSheet("border-radius:10px;padding:8px;font-family:Bahnschrift SemiBold;font-size:14pt;color:green")
		
		self.armature_lbl =QLabel(self)
		self.armature_lbl.setText("Armature")
		self.armature_edit = QLineEdit(self)
		self.armature_edit.setText("Armature")

		self.left_arm_lbl =QLabel(self)
		self.left_arm_lbl.setText("Left Arm")
		self.left_arm_edit = QLineEdit(self)
		self.left_arm_edit.setText("mixamorig:LeftArm")

		self.left_forearm_lbl =QLabel(self)
		self.left_forearm_lbl.setText("Left ForeArm")
		self.left_forearm_edit = QLineEdit(self)
		self.left_forearm_edit.setText("mixamorig:LeftForeArm")

		self.left_wrist_lbl =QLabel(self)
		self.left_wrist_lbl.setText("Left Wrist")
		self.left_wrist_edit = QLineEdit(self)
		self.left_wrist_edit.setText("mixamorig:LeftWrist")

		self.right_arm_lbl =QLabel(self)
		self.right_arm_lbl.setText("Right Arm")
		self.right_arm_edit = QLineEdit(self)
		self.right_arm_edit.setText("mixamorig:RightArm")

		self.right_forearm_lbl =QLabel(self)
		self.right_forearm_lbl.setText("Right ForeArm")
		self.right_forearm_edit = QLineEdit(self)
		self.right_forearm_edit.setText("mixamorig:RightForeArm")

		self.right_wrist_lbl =QLabel(self)
		self.right_wrist_lbl.setText("Right Wrist")
		self.right_wrist_edit = QLineEdit(self)
		self.right_wrist_edit.setText("mixamorig:RightWrist")

		self.left_thigh_lbl =QLabel(self)
		self.left_thigh_lbl.setText("Left Thigh")
		self.left_thigh_edit = QLineEdit(self)
		self.left_thigh_edit.setText("mixamorig:LeftUpLeg")

		self.left_leg_lbl =QLabel(self)
		self.left_leg_lbl.setText("Left Leg")
		self.left_leg_edit = QLineEdit(self)
		self.left_leg_edit.setText("mixamorig:LeftLeg")

		self.left_ankle_lbl =QLabel(self)
		self.left_ankle_lbl.setText("Left Ankle")
		self.left_ankle_edit = QLineEdit(self)
		self.left_ankle_edit.setText("mixamorig:LeftAnkle")

		self.right_thigh_lbl =QLabel(self)
		self.right_thigh_lbl.setText("Right Thigh")
		self.right_thigh_edit = QLineEdit(self)
		self.right_thigh_edit.setText("mixamorig:RightUpLeg")

		self.right_leg_lbl =QLabel(self)
		self.right_leg_lbl.setText("Right Leg")
		self.right_leg_edit = QLineEdit(self)
		self.right_leg_edit.setText("mixamorig:RightLeg")

		self.right_ankle_lbl =QLabel(self)
		self.right_ankle_lbl.setText("Right Ankle")
		self.right_ankle_edit = QLineEdit(self)
		self.right_ankle_edit.setText("mixamorig:RightAnkle")

		self.choose_model_btn = QPushButton(self)
		self.choose_model_btn.setText("Choose 3D Model")
		self.choose_model_btn.clicked.connect(self.on_choose_model_btn_click)
		self.transfer_data_btn = QPushButton(self)
		self.transfer_data_btn.setText("Transfer Data")
		self.transfer_data_btn.clicked.connect(self.on_transfer_data_btn_click)

		self.label_list = [self.armature_lbl,self.left_arm_lbl,self.left_forearm_lbl,self.left_wrist_lbl,
							self.right_arm_lbl,self.right_forearm_lbl,self.right_wrist_lbl,
							self.left_thigh_lbl,self.left_leg_lbl,self.left_ankle_lbl,
							self.right_thigh_lbl,self.right_leg_lbl,self.right_ankle_lbl]
		self.edit_list = [self.armature_edit,self.left_arm_edit,self.left_forearm_edit,self.left_wrist_edit,
							self.right_arm_edit,self.right_forearm_edit,self.right_wrist_edit,
							self.left_thigh_edit,self.left_leg_edit,self.left_ankle_edit,
							self.right_thigh_edit,self.right_leg_edit,self.right_ankle_edit]

		self.button_list = [self.choose_model_btn,self.transfer_data_btn]
		self.format_labels()
		self.format_edit()
		self.format_buttons()

class Controller():
	def __init__(self):
		self.current_screen = "home"
		self.input_source = "image"
		self.image_path = os.getcwd()
		self.video_path = os.getcwd()
		self.character_path = ""
		self.scanned = False

		self.pose = ""
		self.model = "COCO"
		self.device = "CPU"
		self.camera = ""

	def show_home(self):
		self.home_screen = HomeScreen()
		self.home_screen.switch_to_home.connect(self.show_home)
		self.home_screen.switch_to_input.connect(self.show_input)
		self.home_screen.switch_to_network.connect(self.show_network)
		self.home_screen.switch_to_capture.connect(self.show_capture)
		self.home_screen.switch_to_transfer.connect(self.show_transfer)
		self.home_screen.show()

	def show_input(self):
		self.input_screen = InputScreen()
		self.input_screen.switch_to_home.connect(self.show_home)
		self.input_screen.switch_to_input.connect(self.show_input)
		self.input_screen.switch_to_network.connect(self.show_network)
		self.input_screen.switch_to_capture.connect(self.show_capture)
		self.input_screen.switch_to_transfer.connect(self.show_transfer)
		self.input_screen.show()

	def show_network(self):
		self.network_screen = NetworkScreen()
		self.network_screen.switch_to_home.connect(self.show_home)
		self.network_screen.switch_to_input.connect(self.show_input)
		self.network_screen.switch_to_network.connect(self.show_network)
		self.network_screen.switch_to_capture.connect(self.show_capture)
		self.network_screen.switch_to_transfer.connect(self.show_transfer)
		self.network_screen.show()

	def show_capture(self):
		self.capture_screen = CaptureScreen()
		self.capture_screen.switch_to_home.connect(self.show_home)
		self.capture_screen.switch_to_input.connect(self.show_input)
		self.capture_screen.switch_to_network.connect(self.show_network)
		self.capture_screen.switch_to_capture.connect(self.show_capture)
		self.capture_screen.switch_to_transfer.connect(self.show_transfer)
		self.capture_screen.show()

	def show_transfer(self):
		self.transfer_screen = TransferScreen()
		self.transfer_screen.switch_to_home.connect(self.show_home)
		self.transfer_screen.switch_to_input.connect(self.show_input)
		self.transfer_screen.switch_to_network.connect(self.show_network)
		self.transfer_screen.switch_to_capture.connect(self.show_capture)
		self.transfer_screen.switch_to_transfer.connect(self.show_transfer)
		self.transfer_screen.show()


if __name__ == '__main__':
	app = QApplication(sys.argv)
	app.setStyle('Fusion')
	controller = Controller()
	controller.show_home()
	sys.exit(app.exec_())