import sys
import os
import matplotlib
import matplotlib.pyplot as plt
import copy
import tensorflow as tf
import cv2
import imutils
import numpy as np
from sklearn.metrics import pairwise
import pyttsx3
import os
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer=LancasterStemmer()
import numpy
import tflearn
import tensorflow
import random 
import json
import pickle
import speech_recognition  as  sr
import webbrowser as wb 
import subprocess
from subprocess import call
import datetime
from datetime import date
import time

time1= datetime.datetime.now()
date = datetime.date.today()

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

engine = pyttsx3.init('espeak')
voices = engine.getProperty('voices')
print(voices[11].id)  
engine.setProperty('voices', voices[11].id)


path="/usr/bin/firefox %s"
url="https://www.google.com/search?q="
r=sr.Recognizer()

with open("intents2.json") as file:
	data=json.load(file)

try:
	with open("data.pickle","rb") as f:
		words,labels,training,output=pickle.load(f)

except:
	words=[]
	labels=[]
	docs_x=[]
	docs_y=[]

	for intent in data["intents"]:
		for pattern in intent["patterns"]:
			wrds=nltk.word_tokenize(pattern)
			words.extend(wrds)
			docs_x.append(wrds)
			docs_y.append(intent["tag"])


			if intent["tag"] not in labels:
				labels.append(intent["tag"])

	words=[stemmer.stem(w.lower())for w in words if w !="?"]
	words=sorted(list(set(words)))

	labels=sorted(labels)

	training=[]
	output=[]

	out_empty=[0 for _ in range (len(labels))]

	for x,doc in enumerate(docs_x):
		bag=[]

		wrds=[stemmer.stem(w)for w in doc ]

		for w in words:
			if w in wrds:
				bag.append(1)
			else:
				bag.append(0)

		output_row=out_empty[:]
		output_row[labels.index(docs_y[x])]=1

		training.append(bag)
		output.append(output_row)

	training=numpy.array(training)
	output=numpy.array(output)
	

	with open("data.pickle","wb")as f:
		pickle.dump((words,labels,training,output),f)


tensorflow.reset_default_graph()
net=tflearn.input_data(shape=[None,len(training[0])])
net=tflearn.fully_connected(net,34)
net=tflearn.fully_connected(net,34)
net=tflearn.fully_connected(net,len(output[0]),activation="softmax")
net=tflearn.regression(net)

model=tflearn.DNN(net)

try:
	model.load("model.tflearn")
except:
	model.fit(training,output,n_epoch=2500,batch_size=34,show_metric=True)
	model.save("model.tflearn")

def bag_of_words(s,words):
	bag=[0 for _ in range(len(words))]

	s_words=nltk.word_tokenize(s)
	s_words=[stemmer.stem(word.lower()) for word in s_words]

	for se in s_words:
		for i,w in enumerate(words):
			if w==se:
				bag[i]=1

	return numpy.array(bag)

def speak(audio):
	engine.say(audio)
	print("Percy:",audio)
	engine.runAndWait()


def speak1():
	
	speak("Start talking,say quit to stop!")
	while True:
		r = sr.Recognizer()

		with  sr.Microphone()  as  source:
			r.adjust_for_ambient_noise(source)
			audio = r.listen(source)

			try:
				inp = r.recognize_google(audio)
				print("you:",inp)
			except  sr.UnknownValueError:
				speak("error,speak again")
				continue
			except  sr.RequestError  as e:
				print('failed'.format(e))

		if inp == "quit":
			break

		results=model.predict([bag_of_words(inp,words)])[0]
		results_index=numpy.argmax(results)
		tag=labels[results_index]

		if results[results_index]>0.7:
			for tg in data["intents"]:
				if tg['tag']==tag:
					responses=tg['responses']

			check=random.choice(responses)

			if check =="facial emotion recognition":
				subprocess.call("./trial.sh")
				continue

			elif check =="speech emotion recognition":
				subprocess.call("./trial1.sh")
				continue

			elif check == "joke1":

				speak("knock-knock")
				r2=sr.Recognizer()
				with sr.Microphone() as source:
					r2.adjust_for_ambient_noise(source)
					audio = r2.listen(source)

				speak("It's Siri")

				r3=sr.Recognizer()
				with sr.Microphone() as source:
					r3.adjust_for_ambient_noise(source)
					audio = r3.listen(source)

				speak("My thoughts exactly :)")


			elif check =="joke2":

				speak("knock-knock")
				r4=sr.Recognizer()
				with sr.Microphone() as source:
					r4.adjust_for_ambient_noise(source)
					audio = r4.listen(source)

				speak("Boo")

				r5=sr.Recognizer()
				with sr.Microphone() as source:
					r5.adjust_for_ambient_noise(source)
					audio = r3.listen(source)

				speak("No need to cry, it’s only a joke :)")
				

			elif check == "time":
				speak("Right now? Time you got a watch, ha ha! Here in India it is:")
				print(date.strftime('%A'), date.strftime('%d'), date.strftime('%B'), date.strftime('%Y'), time1.strftime('%H:%M'))

			elif check == "news":
				speak("Here are the news pages from the BBC. Everytime I read the news, it's always bad news...")
				time.sleep(2)
				wb.get(path).open_new("https://www.bbc.com/news")
				speak("Press Enter to continue")
				delay=input()

			else:
				speak(check)

		else:
			r6=sr.Recognizer()
			with  sr.Microphone()  as  source:
				speak("I'm not sure i understand.Do you want me to search the internet?[yes/no]")
				time.sleep(2)
				r6.adjust_for_ambient_noise(source)
				audio = r6.listen(source)

				
				try:
					answer=r6.recognize_google(audio)
					print("you:",answer)
				
				except  sr.UnknownValueError:
					speak("error,speak again")
					continue
				except  sr.RequestError  as e:
					print('failed'.format(e))
				

				if(answer=="yes"):
					try:
						wb.get(path).open_new(url+inp)
						speak("Press Enter to continue")
						delay=input()
					except  sr.RequestError  as e:
						print('failed'.format(e))

				else:
					continue


def chat():

	print("Start talking(type quit to stop)!")
	while True:
		inp=input("You: ")
		if inp.lower()=="quit":
			break

		results=model.predict([bag_of_words(inp,words)])[0]
		results_index=numpy.argmax(results)
		tag=labels[results_index]

		if results[results_index]>0.7:
			for tg in data["intents"]:
				if tg['tag']==tag:
					responses=tg['responses']

			check=random.choice(responses)

			if check =="facial emotion recognition":
				subprocess.call("./trial.sh")
				continue

			elif check =="speech emotion recognition":
				from subprocess import call
				subprocess.call("./trial1.sh")
				continue

			elif check =="joke1":
				print("Percy: knock-knock")
				inp1=input("you: ")
				print("Percy: It's Siri")
				inp2=input("you: ")
				print("Percy: My thoughts exactly :)")


			elif check =="joke2":

				print("Percy: knock-knock")
				inp1=input("you: ")
				print("Percy: Boo")
				inp2=input("you: ")
				print("Percy: No need to cry, it’s only a joke:)")

			elif check =="time":
				print("Percy: Right now? Time you got a watch, ha ha! Here in India it is:")
				print(date.strftime('%A'), date.strftime('%d'), date.strftime('%B'), date.strftime('%Y'), time1.strftime('%H:%M'))

			elif check =="news":
				print("Percy: Here are the news pages from the BBC. Everytime I read the news, it's always bad news...")
				time.sleep(2)
				wb.get(path).open_new("https://www.bbc.com/news")

			else:
				print("Percy:",check)

						
		else:
			answer=input("Percy: I'm not sure i understand.Do you want me to search the internet?[y/n]")
			if(answer=="y"):
				try:
					wb.get(path).open_new(url+inp)
				except  sr.RequestError  as e:
					print('failed'.format(e))

			else:
				continue


def gesture():

	os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
	import tensorflow as tf
	sequence=""
	def predict(image_data):
		sess = tf.Session()
		predictions = sess.run(softmax_tensor, \
			{'DecodeJpeg/contents:0': image_data})
		top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
		max_score = 0.0
		res = ''
		for node_id in top_k:
			human_string = label_lines[node_id]
			score = predictions[0][node_id]
			if score > max_score:
				max_score = score
				res = human_string
		return res, max_score

	# Loads label file, strips off carriage return
	label_lines = [line.rstrip() for line
	                   in tf.gfile.GFile("logs/trained_labels.txt")]

	# Unpersists graph from file
	with tf.gfile.FastGFile("logs/trained_graph.pb", 'rb') as f:
	    graph_def = tf.GraphDef()
	    graph_def.ParseFromString(f.read())
	    _ = tf.import_graph_def(graph_def, name='')

	with tf.Session() as sess:
	    # Feed the image_data as input to the graph and get first prediction
	    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

	    c = 0

	    cap = cv2.VideoCapture(-1)

	    res, score = '', 0.0
	    i = 0
	    mem = ''
	    consecutive = 0
	    sequence = ''
	    
	    while True:
	        ret, img = cap.read()
	        img = cv2.flip(img, 1)
	        
	        if ret:
	            x1, y1, x2, y2 = 100, 100, 300, 300
	            img_cropped = img[y1:y2, x1:x2]

	            c += 1
	            image_data = cv2.imencode('.jpg', img_cropped)[1].tostring()
	            
	            a = cv2.waitKey(1) # waits to see if `esc` is pressed
	            
	            if i == 4:
	                res_tmp, score = predict(image_data)
	                res = res_tmp
	                i = 0
	                if mem == res:
	                    consecutive += 1
	                else:
	                    consecutive = 0
	                if consecutive == 2 and res not in ['nothing']:
	                    if res == 'space':
	                        sequence += ' '
	                    elif res == 'del':
	                        sequence = sequence[:-1]
	                    else:
	                        sequence += res
	                    consecutive = 0
	            i += 1
	            cv2.putText(img, '%s' % (res.upper()), (100,400), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,255,255), 4)
	           
	            cv2.putText(img, '(score = %.5f)' % (float(score)), (100,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
	            mem = res
	            cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)
	            cv2.imshow("img", img)
	            img_sequence = np.zeros((200,1200,3), np.uint8)
	            cv2.putText(img_sequence, '%s' % (sequence.upper()), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
	            
	            cv2.imshow('sequence', img_sequence)
	            
	            if a == 27: # when `esc` is pressed
	                break
	
	      
	          
	cap.release()
	cv2.destroyAllWindows()
	print("you: ",sequence) 
	inp=sequence

	results=model.predict([bag_of_words(inp,words)])[0]
	results_index=numpy.argmax(results)
	tag=labels[results_index]

	if results[results_index]>0.7:
		for tg in data["intents"]:
			if tg['tag']==tag:
				responses=tg['responses']

		check = random.choice(responses)

		if check == "facial emotion recognition":
			subprocess.call("./trial.sh")
			
			

		elif check =="speech emotion recognition":
			subprocess.call("./trial1.sh")
			


		elif check == "joke1":

			print("Percy: knock-knock")
			inp1=input("you: ")
			print("Percy: It's Siri")
			inp2=input("you: ")
			print("Percy: My thoughts exactly :)")


		elif check =="joke2":

			print("Percy: knock-knock")
			inp1=input("you: ")
			print("Percy: Boo")
			inp2=input("you: ")
			print("Percy: No need to cry, it’s only a joke:)")

		elif check == "time":
			print("Percy: Right now?Time you got a watch, ha ha! Here in India it is:")
			print(date.strftime('%A'), date.strftime('%d'), date.strftime('%B'), date.strftime('%Y'), time1.strftime('%H:%M'))

		elif check =="news":
			print("Percy: Here are the news pages from the BBC. Everytime I read the news, it's always bad news...")
			time.sleep(2)
			wb.get(path).open_new("https://www.bbc.com/news")

		else:
			print("Percy:",check)

	else:
				
		print("Percy: I'm not sure i understand.Do you want me to search the internet[y/n]?")
		answer=input()
					
		if(answer=="y"):
			try:
				wb.get(path).open_new(url+inp)
			except  sr.RequestError  as e:
				print('failed'.format(e))
		

if __name__=="__main__":
	while True:
		reply=input("Do you want to type,speak or use sign language(type quit to stop)? ")
		if reply=="type":
			chat()
		elif reply=="speak":
			speak1()
		elif reply=="sign language":
			while True:
				ask=input("Percy: Press Enter to continue and type'quit'to stop ")
				if ask=="quit":
					break
				gesture()

		elif reply=="quit":
			break

		else :
			print("please enter a valid option")
			continue



	




