## This sciprt adds LFW faces to Microsoft Face API database, and enables us to test the performance of Face API recognition



import http.client, urllib.request, urllib.parse, urllib.error, base64
import json, codecs
import matplotlib.pyplot as plt
import cv2
# from os import listdir, makedirs, remove, rename
# from os.path import join, exists, isdir
import time
import numpy as np

from sklearn.datasets import fetch_lfw_people

#fetch lfw data 
lfw_people = fetch_lfw_people(min_faces_per_person = 15, resize = None)
print('end')
#print(len(lfw_people.target_names))
#print(target_names)
print(len(lfw_people.images))
print(len(lfw_people.target))
print(len(lfw_people.target_names))
print(len(lfw_people.binary))

print('Start!')
# testing data structure
# n=1
# print(lfw_people.target[n])
# print(lfw_people.target_names[lfw_people.target[n]])
# plt.imshow(lfw_people.images[n])
# plt.show()

# Add individual person and create dictionary

# Change this to change person group. innovlabv1 or lfwtest
groupId = 'innovlabv1'

headers_1 = {
    # Request headers
    'Content-Type': 'application/json',
    'Ocp-Apim-Subscription-Key': '486688345a164773bf6fb69d6466fd04',
}

headers_2 = {
    # Request headers
    'Content-Type': 'application/octet-stream',
    'Ocp-Apim-Subscription-Key': '486688345a164773bf6fb69d6466fd04',
}


params_1 = urllib.parse.urlencode({
})


## Index individual name
nameId = {}
cntId = {}
cnt = 0
for name in lfw_people.target_names:
	if cnt == 19:
		time.sleep(60)
		cnt = 0
		print('20call')

	cnt = cnt + 1
	body_1 = {
		'name' : '%s' %name
	} 

	try:
		conn = http.client.HTTPSConnection('westcentralus.api.cognitive.microsoft.com')
		conn.request("POST", "/face/v1.0/persongroups/%s/persons?%s" % (groupId, params_1), str(body_1), headers_1)
		response = conn.getresponse()
		data = response.read().decode()
		jdata = json.loads(data)
		personId = jdata['personId']

		nameId[name] = personId
		cntId[name] = 3

		print(name, ':', personId)
		conn.close()

	except Exception as e:
		print("[Errno {0}] {1}".format(e.errno, e.strerror))

print('Finished adding persons')
    #acquire the index of this person name in target_name list, in order to find the person's face 
    # target_idx = target_names.index(name)
    # face_idx = target.index()
# Name ID dictionary finished. Add LFW face into Face API
# Limit the input face per person as 3
for i in range(len(lfw_people.images)):
	if cnt == 19:
		time.sleep(60)
		cnt = 0
		print('20call')
	
	face = lfw_people.images[i]
	body_2 = lfw_people.binary[i]
	nameStr = lfw_people.target_names[lfw_people.target[i]]
	currId = nameId[nameStr]

	if cntId[nameStr] >= 0:
		cnt = cnt + 1
		params_2 = urllib.parse.urlencode({
	    	# Request parameters
	    	'personGroupId' : '%s' %groupId,
	    	'personId' : '%s' %currId ,
	    	#'userData': '{string}',
	    	#'targetFace': '{string}',
		})
		#body = open(KEY,'rb').read()

		try:
			conn = http.client.HTTPSConnection('westcentralus.api.cognitive.microsoft.com')
			conn.request("POST", "/face/v1.0/persongroups/%s/persons/%s/persistedFaces?%s" %(groupId, currId ,params_2) , body_2, headers_2)
			response = conn.getresponse()
			data = response.read()
			print(data)
			conn.close()
		except Exception as e:
			print("[Errno {0}] {1}".format(e.errno, e.strerror))
		# FIGURE A WAY TO CONVERT BINARY ARRAY TO IMAGE AND THEN IMREAD

	cntId[nameStr] = cntId[nameStr] - 1

print('Finished adding faces')

headers_3 = {
    # Request headers
    'Ocp-Apim-Subscription-Key': '486688345a164773bf6fb69d6466fd04',
}

params_3 = urllib.parse.urlencode({
})
body_3 = {}
try:
    conn = http.client.HTTPSConnection('westcentralus.api.cognitive.microsoft.com')
    conn.request("POST", "/face/v1.0/persongroups/%s/train?%s" %(groupId, params_3), str(body_3), headers_3)
    response = conn.getresponse()
    data = response.read()
    print(data)
    conn.close()
except Exception as e:
    print("[Errno {0}] {1}".format(e.errno, e.strerror))
#Dict for lfwtest:Abdullah Gul : 75ac4b4f-91ef-461a-a0f6-1fc3324d09cb
# Alejandro Toledo : bf7b0f98-05f9-4bcb-9b77-fc3f5e7a33f0
# Alvaro Uribe : f19ad52a-fbe9-41fb-bc03-68007a79c6c0
# Amelie Mauresmo : b44a3327-1d9d-4bf5-9850-9852b29b2c23
# Andre Agassi : 1512189d-2947-4fd5-9e47-1d0a428a1dc9
# Andy Roddick : 2ebcdff3-432e-464a-af21-8a0563367373
# Angelina Jolie : a3e04b1a-4ce3-4d4e-8e4d-8ff65ea84711
# Ariel Sharon : f3432f6a-22b0-4579-9c2c-07b33b09cf5a
# Arnold Schwarzenegger : cc643d8b-3cc1-48de-bd8e-00e9b34d3ebe
# Atal Bihari Vajpayee : e4021be7-4cf0-4233-8a5e-e1a155357df0
# Bill Clinton : 0951a568-14a9-4fb8-ad06-5cd6701a2377
# Bill Gates : 013ab0f0-0943-4db9-b0a4-d9912de39dfd
# Bill Simon : 42f821f9-d1e2-4f88-a15b-3f9665253ab0
# Carlos Menem : e5a412c7-7581-461b-9d6c-4eba6883e19e
# Carlos Moya : 639b3026-e8f9-4276-9eec-aaeea5c50f6a
# Colin Powell : d5c27c84-6f08-443c-a592-ba02121aa236
# David Beckham : 115624eb-de6a-481a-a599-11e7d00a547d
# Dominique de Villepin : 9785c9a5-cc26-4ce5-95f0-44ccc1469af1
# Donald Rumsfeld : b4f6225e-93fb-428b-ad56-dd5e370170d6
# 20call
# Fidel Castro : f6067bbe-36ad-4a3e-9cb1-3cf74357c8cb
# George Robertson : 89211d63-63b2-4986-9e6d-2d7c8972dd20
# George W Bush : 2f2f18d6-9993-404e-895d-a917b7e26f99
# Gerhard Schroeder : 0d59cbc4-a6a6-44c4-82fd-d473647c73b6
# Gloria Macapagal Arroyo : bf316d61-4122-4296-8020-021b8b843e52
# Gray Davis : a0592a2d-33e5-4e80-84cf-e569aedd2509
# Guillermo Coria : 8f95e94b-58bc-49e8-ac9b-1396469a230a
# Halle Berry : 4aaedf8c-962f-4437-8012-ae143805353c
# Hamid Karzai : d3767dbd-0531-40a0-9256-e0b7e2e21fed
# Hans Blix : 01b382c4-d336-47a8-8b4c-dce99c342484
# Hu Jintao : 87c79ded-319d-4212-ad73-0a861233cd36
# Hugo Chavez : c8e64ea5-ed0d-48f1-a32f-86cfe690e5eb
# Igor Ivanov : 9a3a9103-5bb6-49a3-8d3a-5f2a257f2be4
# Jack Straw : f91073e2-a370-4c36-972e-fee4e5aa2baf
# Jacques Chirac : c1133d21-0dc1-4cd1-8101-7cc22b2ccfe1
# Jean Charest : 731e7f48-130e-407c-9c43-36b9064eec68
# Jean Chretien : c2ae19cf-8049-42e6-864e-04a341da3309
# Jennifer Aniston : cb499bd3-3b87-4ac0-8197-656ae1c08add
# Jennifer Capriati : 4a1d5661-c472-4c59-ab04-e2fd9c3c2eb2
# 20call
# Jennifer Lopez : 0c904325-8461-46c0-b92b-13ffcbaaf8f4
# Jeremy Greenstock : 91215983-d293-469d-8c32-c9bcc9b80c51
# Jiang Zemin : 39654188-d6b6-4a19-a4db-811dcc14b62e
# John Ashcroft : ba84e3db-fd6a-4470-9393-94609b19da3a
# John Bolton : 3fa71f17-e38b-458c-ab56-fb1bf4889e39
# John Howard : ed8bc5ef-5f0e-48f7-a4ec-069b946fca1c
# John Kerry : ecc28df2-b9f7-4abf-b846-5d9137d8a668
# John Negroponte : 611a6933-889a-4d4a-a644-cfa229007474
# John Snow : d0c1f86a-c49f-4032-8b9e-4d48242469c6
# Joschka Fischer : f3c57265-784c-4dbe-8842-56c2fa1250fd
# Jose Maria Aznar : 61de3f15-b109-4035-ac4c-ec69f1d9250d
# Juan Carlos Ferrero : 3ed3f616-aec1-4307-b8eb-558182953aca
# Julianne Moore : 176ce76a-299a-4a90-bec8-4996a878e5e9
# Julie Gerberding : d2e47520-4d3a-46e8-a38f-4f095b99771f
# Junichiro Koizumi : 65d87b49-dda5-4fd9-afd5-26583fa06cfa
# Kofi Annan : a3596e21-e3e2-4da1-8dce-d9fe4c366f3f
# Lance Armstrong : 86dfeabc-b3d4-43fa-820f-6a3fd13b8c95
# Laura Bush : 4e7cf456-ab9a-45e0-935b-360f019fb2de
# Lindsay Davenport : 42835faf-2ada-4a74-984d-aad66b8d9211
# 20call
# Lleyton Hewitt : 588b40d1-115e-4368-bfd0-2e91ed7c5b8f
# Luiz Inacio Lula da Silva : ab16cd5f-f97d-408c-8cdd-29f7d9d9b6a7
# Mahmoud Abbas : 732f2975-38a1-4c65-88fe-dc9076f835a2
# Megawati Sukarnoputri : 3d4e9836-f2c7-4f0f-9d2b-6bd5447f646e
# Meryl Streep : 9f6927f0-a664-43b2-8b73-da9a7e06ee5b
# Michael Bloomberg : 1b555402-1c16-4518-aee1-9655160ba92e
# Michael Schumacher : bf5af7e9-b295-4297-90be-98897f1ce6e9
# Mohammed Al-Douri : 0591b8ba-366f-4414-a3ee-e2d4cfafe9ac
# Nancy Pelosi : 0c7fc016-bdcd-4756-af8e-67359622e8af
# Naomi Watts : 481bdf2f-68c4-4509-8916-cfabe842bc68
# Nestor Kirchner : 59ee5212-33a4-4180-a275-5421583332d4
# Nicole Kidman : 734e987f-ab5c-4644-b478-d744513b8710
# Norah Jones : f2443c00-76a2-441a-a116-92615b7edece
# Paul Bremer : ac0f31a7-ba55-4ec6-887a-3815ff97bbdc
# Pervez Musharraf : 6309100d-3336-4dc9-a739-bd5304a430d5
# Pete Sampras : 37d0b6a0-d87b-4cca-80aa-8276b0a7d10e
# Pierce Brosnan : 96bb0cbf-9099-4a6f-b5df-8402e68b92ce
# Recep Tayyip Erdogan : a4c38a0e-f78e-4577-82f5-ef4eeb784f72
# Renee Zellweger : 6d087810-3709-4a9b-9373-d9f94033ffb9
# 20call
# Ricardo Lagos : ca20aefc-213f-40d6-879b-1dad1fd57350
# Richard Myers : 88de012e-088b-431f-8078-c1d56dd690ee
# Roh Moo-hyun : 67864bab-3b48-402a-ab12-1d3dc6b79f69
# Rudolph Giuliani : f3f68682-f0a9-4ce5-939e-049360255726
# Saddam Hussein : 3ed73e33-cff5-4e42-acc2-17ec5f5814ea
# Serena Williams : ef773dad-9a18-43f9-b45f-acbb03e27fb3
# Silvio Berlusconi : 6727a6ef-85fc-43cb-9bee-3a852e5fe477
# Spencer Abraham : f0fe7cb6-c80d-43f7-8054-c111380291e1
# Taha Yassin Ramadan : 683253f2-59e1-4d37-9277-6e6ca06988a7
# Tiger Woods : f8516926-8805-4cb4-acc1-e8fa8f1a22b4
# Tim Henman : deda933d-ea8c-442d-994e-3be16f05f7bb
# Tom Daschle : 4c6d7f28-589b-49c5-ac8d-f219a2992c5a
# Tom Ridge : 8327e112-5ad7-48ba-b014-c722813b74d2
# Tommy Franks : c1c5cacf-564b-4b7b-860c-72621d8b510c
# Tony Blair : 77668a75-3c1b-4282-94e6-c2144ab7537f
# Trent Lott : b9f009ea-db67-476e-bd62-341d29f106ff
# Venus Williams : fa10611d-fbd1-4ece-b79d-f425f55667c8
# Vicente Fox : 4c70944d-3ea6-4de0-a9de-3bf3877ffad7
# Vladimir Putin : 4ee05579-7be8-4c02-a196-0d08f146a22e
# 20call
# Winona Ryder : 6e162d1c-0975-4c39-acf7-c7d92ded0361