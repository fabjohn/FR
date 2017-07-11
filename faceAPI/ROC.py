##This script creates an test person group for ROC evaluation

import http.client, urllib.request, urllib.parse, urllib.error, base64
import json, codecs
import matplotlib.pyplot as plt
import cv2 
import time
import numpy as np
from sklearn.datasets import fetch_lfw_people
import os
import sys


def main():
	key = '7a620905befc45e5adbfef7b16b56b01'
	##Fetech data from LFW dataset. 
	person_size = 100	##5 times
	fold_n = int(person_size / 5)
	print(fold_n)
	individual_size = 2
	z=9
	lfw_people = fetch_lfw_people(min_faces_per_person = 10, resize = None)
	print(len(lfw_people.images))
	#print(len(lfw_people.target))
	print(len(lfw_people.target_names))
	print(len(lfw_people.binary))

	train_x = [] ##label
	train_y = [] ##binary data
	test_x = []	##test labels
	test_y = []	##test binary data
	temp = [0]*5000
	for i in range(len(lfw_people.target)):
		if lfw_people.target[i] < person_size:
			if temp[lfw_people.target[i]] <1:	
				train_x.append(lfw_people.target_names[lfw_people.target[i]])
				train_y.append(lfw_people.binary[i])
			elif 0<temp[lfw_people.target[i]]<individual_size:
				test_x.append(lfw_people.target_names[lfw_people.target[i]])
				test_y.append(lfw_people.binary[i])
			temp[lfw_people.target[i]] += 1


	print(train_x)
	print(len(train_x))
	print('Let the fun begin!')
	#print(train_y)
	# faceId = detectFace(test_y[0], key)
	# x, y, z = identifyFace(faceId, key, '1test1')
	# print(x)
	# print(y)
	# print(z)

	# start training request
	# first create person group for first 5 persons
	cnt = 0
	nameId = {}
	for i in range(fold_n):
		groupId = '%stest%s'%(z,i)
		createPersonGroups(groupId, key)

		## Create persons 5 per group
		for j in range(5):

			curr_label = train_x[j+cnt]
			personId = createPerson(groupId, key, curr_label) 		##possibility to fail due to multiple face. Need face crop
			nameId[curr_label] = personId
			## Load faces into the dataset
			curr_binary = train_y[j+cnt]
			addPersonFace(groupId, key, personId, curr_binary)

		##train person groups
		trainPersonGroup(groupId, key)
		cnt += 5

	predictList = []
	confidenceLists = []
	for item in test_y:
		faceId = detectFace(item, key)
		print(faceId)
		maxConf = 0
		temp = []
		for i in range(fold_n):
			groupId = '%stest%s'%(z,i)
			bestPId, bestConf, confList  = identifyFace(faceId, key, groupId)
			if bestConf > maxConf:
				predictId = bestPId
				maxConf = bestConf
			temp = temp + confList 		##Merge confidence lists

		predictList.append(predictId)
		confidenceLists.append(temp)

	if len(predictList) != len(test_x):
		print('this is not right. fuck')
	cnt = 0
	for i in range(len(predictList)):
		if predictList[i] == nameId[test_x[i]]:
			cnt +=1
	accuracy = cnt/len(predictList)
	print(accuracy)
	# print(len(predictList))
	# print(predictList)
	#print(confidenceLists)
	##genereate ROC curve
	generateRoc(predictList, test_x, confidenceLists, nameId)




def generateRoc(predictList,test_data, confidenceLists, nameId):
	fprs = []
	tprs = []
	for thrshld in range(101):
		tp = tn = fp = fn = 0
		thrshld = thrshld / 100
		for i in range(len(predictList)):
			curr_Conf = confidenceLists[i]
			sum_true = sum(i >= thrshld for i in curr_Conf)
			predict_same = predictList[i] == nameId[test_data[i]]
			if sum_true>0 and predict_same:
				tp += 1
				fp += sum_true - 1
				tn += len(curr_Conf) - sum_true
			elif sum_true>0 and not predict_same:
				fp += sum_true
				tn += len(curr_Conf) - sum_true
			elif sum_true<= 0 and predict_same:
				fn += 1
				tn += len(curr_Conf) - 1
			elif sum_true<= 0 and not predict_same:
				tn += len(curr_Conf)
		print(fn)
		if tp + fn ==0:
			tpr = 0
		else:
			tpr = float(tp) / float(tp+fn)
		if fp + tn == 0:
			fpr = 0
		else:
			fpr = float(fp) / float(fp+tn)
		fprs.append(fpr)
		tprs.append(tpr)
	print(tprs)
	print(fprs)
	AUC = getAUC(fprs, tprs)
	print(AUC)
	plt.plot(fprs, tprs)
	plt.xlabel('fpr')
	plt.ylabel('tpr')
	plt.title('ROC curve')
	plt.show()

def getAUC(fprs, tprs):
    sortedFprs, sortedTprs = zip(*sorted(zip(*(fprs, tprs))))
    sortedFprs = list(sortedFprs)
    sortedTprs = list(sortedTprs)
    if sortedFprs[-1] != 1.0:
        sortedFprs.append(1.0)
        sortedTprs.append(sortedTprs[-1])
    return np.trapz(sortedTprs, sortedFprs)


def createPersonGroups(groupId, key):
	headers = {
		'Content-Type': 'application/json',
	    'Ocp-Apim-Subscription-Key': '%s'%key,
	}
	body = {
		'name':'%s'%groupId
	}
	params = urllib.parse.urlencode({
	})
	try: 
		#r = requests.post('https://westcentralus.api.cognitive.microsoft.com/face/v1.0/persongroups/innovlabv1', json = body, headers = headers)
		conn = http.client.HTTPSConnection('westus.api.cognitive.microsoft.com')
		conn.request('PUT', '/face/v1.0/persongroups/%s?%s' % (groupId,params), str(body), headers)
		response = conn.getresponse()
		data = response.read()
		print(data)
		conn.close()
		#print(r.json())
	except Exception as e:
		print('[Errno {0}] {1}'.format(e.errno, e.strerror))


def createPerson(groupId, key, name):
	headers = {
	    # Request headers
	    'Content-Type': 'application/json',
	    'Ocp-Apim-Subscription-Key': '%s'%key,
	}
	body = {
		'name' :'%s'%name
	}
	params = urllib.parse.urlencode({
	})
	try:
		conn = http.client.HTTPSConnection('westus.api.cognitive.microsoft.com')
		conn.request("POST", "/face/v1.0/persongroups/%s/persons?%s" % (groupId, params), str(body), headers)
		response = conn.getresponse()
		data = response.read().decode()
		jdata = json.loads(data)
		personId = jdata['personId']
		#nameId[curr] = personId
		#cntId[name] = 3
		print(name, ':', personId)
		conn.close()

	except Exception as e:
		print("[Errno {0}] {1}".format(e.errno, e.strerror))

	return personId


def addPersonFace(groupId, key, personId, binary_data):
	headers = {
	    # Request headers
	    'Content-Type': 'application/octet-stream',
	    'Ocp-Apim-Subscription-Key': '%s'%key,
	}
	params = urllib.parse.urlencode({
    	# Request parameters
    	'personGroupId' : '%s' %groupId,
    	'personId' : '%s' %personId ,
    	#'userData': '{string}',
    	#'targetFace': '{string}',
	})
	try:
		conn = http.client.HTTPSConnection('westus.api.cognitive.microsoft.com')
		conn.request("POST", "/face/v1.0/persongroups/%s/persons/%s/persistedFaces?%s" %(groupId, personId ,params) , binary_data, headers)
		response = conn.getresponse()
		data = response.read()
		print(data)
		conn.close()
	except Exception as e:
		print("[Errno {0}] {1}".format(e.errno, e.strerror))


def trainPersonGroup(groupId, key):
	headers = {
	    # Request headers
	    'Ocp-Apim-Subscription-Key': '7a620905befc45e5adbfef7b16b56b01',
	}
	params = urllib.parse.urlencode({
	})
	body = {}
	try:
	    conn = http.client.HTTPSConnection('westus.api.cognitive.microsoft.com')
	    conn.request("POST", "/face/v1.0/persongroups/%s/train?%s" %(groupId, params), str(body), headers)
	    response = conn.getresponse()
	    data = response.read()
	    print(data)
	    conn.close()
	except Exception as e:
	    print("[Errno {0}] {1}".format(e.errno, e.strerror))		


## Test the datasets and generate ROC curve
def detectFace(binary_data, key):
	headers = {
	    # Request headers
	    'Content-Type': 'application/octet-stream',
	    'Ocp-Apim-Subscription-Key': '%s'%key,       ##your access key
	}
	params = urllib.parse.urlencode({
	    # Request parameters
	    'returnFaceId': 'true',
	    # Request face bounding box parameters
	    'returnFaceLandmarks': 'ture',                  
	    # Change this to enable more detection attributes 
	    'returnFaceAttributes': 'age,gender,glasses',   
	})
	try:
		conn = http.client.HTTPSConnection('westus.api.cognitive.microsoft.com')
		conn.request("POST", "/face/v1.0/detect?%s" % params, binary_data, headers)
		response = conn.getresponse()
		reader = codecs.getreader('utf-8')
		data = response.read().decode()
		#jdata for index json file
		jdata = json.loads(data)
		print(jdata)
		#print(jdata[1])
		#print(jdata[0]['faceRectangle'])
		faceId  = []
		#For loop for display bouding box 
		min_value = 0
		for face in jdata:
			temp = face['faceRectangle']
			#cv2.rectangle(img,(temp['left'],temp['top']),(temp['left']+temp['width'] ,  temp['top']+temp['height']), (0,255,0),2)
			if temp['top']*temp['width'] > min_value:
				min_value = temp['top']*temp['width']						##only the largest face get identified. 
				max_face = face['faceId']
		conn.close()
	except Exception as e:
		print("[Errno {0}] {1}".format(e.errno, e.strerror))

	return max_face


def identifyFace(faceId, key, groupId):
    headers = {
        # Request headers
        'Content-Type': 'application/json',
        'Ocp-Apim-Subscription-Key': '%s'%key
    }
    params = urllib.parse.urlencode({
    })
    body = {
    		'personGroupId' : '%s'%groupId,
            'faceIds':[
                '%s'%faceId
            ],
            'maxNumOfCandidatesReturned': 5,
            'confidenceThreshold' : 0
    }
    confList = []
    n=0
    try:
        conn = http.client.HTTPSConnection('westus.api.cognitive.microsoft.com')
        conn.request("POST", "/face/v1.0/identify?%s" % params, str(body), headers)
        response = conn.getresponse()
        data = response.read().decode()
        jdata = json.loads(data)
        print(jdata)
        if 'candidates' not in jdata[0]:
        	print('shit is not right')
        temp = jdata[0]
        for person in temp['candidates']:
        	if n == 0:
        		mId = person['personId']			##max conf 
        		mConf = person['confidence']
        	confList.append(person['confidence'])   ##only 5 
        	n += 1
#        print(data)
        conn.close()
    except Exception as e:
        print("[Errno {0}] {1}".format(e.errno, e.strerror))

    return mId, mConf, confList


if __name__ == '__main__':
    main()