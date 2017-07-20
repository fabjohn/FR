##This file contains necessary calls for Kairos API
import base64
import requests
import kairos_face
import pickle
from sklearn.datasets import fetch_lfw_people
import numpy as np
import matplotlib.pyplot as plt
import time
import os, os.path

def main():
	headers = {
		'Content-Type' : 'application/json',
		'app_id': 'ead2b29d',
		'app_key': '11c8ccd3a34cdebcb7c340b6041cc587'
	}
	path = 'cropped_data'
	person_size = 1000	##5 times
	individual_size = 2
	z=1
	#lfw_people = fetch_lfw_people(min_faces_per_person = 2, resize = None)
	# print(len(lfw_people.images))
	# #print(len(lfw_people.target))
	# print(len(lfw_people.target_names))
	# print(len(lfw_people.binary))

	target_names = []
	binary_data = []

	for item in os.listdir(path):
		curr_file = os.path.join(path, item)
		if len(os.listdir(curr_file)) > 1:
			item = item.replace('_', ' ')
			#print(item)
			target_names.append(item)
			binary_temp = []
			for face in os.listdir(curr_file):
				face_file = os.path.join(curr_file, face)
				temp = open(face_file, 'rb').read()
				binary_temp.append(temp)
			binary_data.append(binary_temp)


	train_x = [] ##label
	train_y = [] ##binary data
	test_x = []	##test labels
	test_y = []	##test binary data
	temp = [0]*5000
	time_cnt = 0

	for i in range(len(target_names)):
		if i < person_size:
			train_x.append(target_names[i])
			train_y.append(binary_data[i][0])
			
			test_x.append(target_names[i])
			test_y.append(binary_data[i][1])

			#backup_y.append(binary_data[i][2])


	gallery_name = 'lfwtest%s'%z
	# remove(gallery_name, headers)
	# for i in range(len(train_x)):			##traverse through all the training person
	# 	curr_name = train_x[i]
	# 	curr_binary = train_y[i]
	# 	print(curr_name)
	# 	curr_base64 = base64.b64encode(curr_binary).decode('ascii')
	# 	enroll(curr_base64, curr_name, gallery_name, headers)
	# 	time_cnt+=1
	# 	if time_cnt == 60:
	# 		time.sleep(60)
	# 		time_cnt = 0

	print('finished training group!')

	predict_list = []
	confidence_list = []
	for item in test_y:
		item_base64 = base64.b64encode(item).decode('ascii')
		predict_name, predict_conf, conf = recognize(item_base64, gallery_name, headers)
		predict_list.append(predict_name)
		confidence_list.append(conf)
		time_cnt += 1
		if time_cnt == 60:
			time.sleep(60)
			time_cnt = 0
		#print(conf)
	with open('kairos_conf', 'wb') as fp:
		pickle.dump(predict_list, fp)
		pickle.dump(confidence_list, fp)
		pickle.dump(test_x, fp)
	print(predict_list)

	print(test_x)
	cnt = 0
	for i in range(len(predict_list)):
		if predict_list[i] == test_x[i]:
			cnt+=1
	print(cnt/len(predict_list))
	generateRoc(predict_list, test_x, confidence_list)

def generateRoc(predictList,test_data, confidenceLists):
	fprs = []
	tprs = []
	for thrshld in range(101):
		tp = tn = fp = fn = 0
		thrshld = thrshld / 100
		for i in range(len(predictList)):
			curr_Conf = confidenceLists[i]

			sum_true = sum(j >= thrshld for j in curr_Conf)
			#print(sum_true)
			predict_same = predictList[i] == test_data[i]
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

		print('fn',fn)
		print('tp',tp)
		print('fp',fp)
		print('tn',tn)
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


def remove(gallery_name, headers):
	url = 'https://api.kairos.com/gallery/remove'

	values = {
		'gallery_name': '%s'%gallery_name
	}

	r = requests.post(url, json=values, headers=headers)


def enroll(image, person_name, gallery_name, headers):
	url = 'https://api.kairos.com/enroll'

	# fp = open('test.jpg', 'rb') 
	# image = base64.b64encode(fp.read()).decode('ascii')

	values = {
		'image': image,
		'subject_id': '%s'%person_name,
		'gallery_name': '%s'%gallery_name
	}	

	r = requests.post(url, json=values, headers=headers)

	print(r)
	print(r.json())

def recognize(image, gallery_name, headers):
	url = 'https://api.kairos.com/recognize'

	values = {
		'image': image,
		'gallery_name': '%s'%gallery_name,
		'threshold': '0.70',
		'max_num_results':'1000'
	}

	r = requests.post(url, json=values, headers=headers)
	print(r.json())
	request_data = r.json()
	if 'images' in request_data:
		if 'candidates' in request_data['images'][0]:
			n = 0
			conf_list = []
			candidates_list = request_data['images'][0]['candidates']
			for candidate in candidates_list:
				if n == 0:
					predict_name = candidate['subject_id']
					predict_conf = candidate['confidence']
				n = 1
				conf_list.append(candidate['confidence'])
		else:
			predict_name = '0'
			predict_conf = 1
			conf_list = [0]*1000
	else:
		predict_name = '0'
		predict_conf = 1
		conf_list = [0]*1000

	return predict_name, predict_conf, conf_list



def detect(image, headers):
	url = 'https://api.kairos.com/detect'

	values = {
		'image': image,
		'selector': 'ROLL'
	}

	r = requests.post(url, json=values, headers=headers)

	print(r.json())	

if __name__ == '__main__':
    main()


# enroll()
# headers = {
# 	'Content-Type' : 'application/json',
# 	'app_id': 'ead2b29d',
# 	'app_key': '11c8ccd3a34cdebcb7c340b6041cc587'
# }
# fp = open('Junxiao/Junxiao_Yang_0005.jpg', 'rb')
# image = base64.b64encode(fp.read()).decode('ascii')
# enroll(image, 'Junxiao Xyz' ,'Office', headers)
# x ,y,z =recognize(image,'Office', headers)
# kairos_face.settings.app_id = 'ead2b29d'
# kairos_face.settings.app_key = '11c8ccd3a34cdebcb7c340b6041cc587'

# # kairos_face.enroll_face(file = 'test.jpg', subject_id='Joseph', gallery_name='Office')

# galleries_list = kairos_face.get_galleries_names_list()

# gallery_subjects = kairos_face.get_gallery('Office')
# print(gallery_subjects)