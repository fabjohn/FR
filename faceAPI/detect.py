import http.client, urllib.request, urllib.parse, urllib.error, base64
import json, codecs
import matplotlib.pyplot as plt
import cv2 
KEY = 'Test/test3.jpg'

headers = {
    # Request headers
    'Content-Type': 'application/octet-stream',
    'Ocp-Apim-Subscription-Key': '486688345a164773bf6fb69d6466fd04',       ##your access key
}

params = urllib.parse.urlencode({
    # Request parameters
    'returnFaceId': 'true',
    # Request face bounding box parameters
    'returnFaceLandmarks': 'ture',                  
    # Change this to enable more detection attributes 
    'returnFaceAttributes': 'age,gender,glasses',   
})

#body for http request
body = open(KEY,'rb').read()

#img read for display bounding box
img = cv2.imread(KEY)
# plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
# plt.show()
# body = {
#     'url' : 'http://i.imgur.com/8b7ZxE3.jpg'
# }


try:
    conn = http.client.HTTPSConnection('westcentralus.api.cognitive.microsoft.com')
    conn.request("POST", "/face/v1.0/detect?%s" % params, body, headers)
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
    for face in jdata:

        temp = face['faceRectangle']
        cv2.rectangle(img,(temp['left'],temp['top']),(temp['left']+temp['width'] ,  temp['top']+temp['height']), (0,255,0),2)
        faceId.append(face['faceId'])


    #print(data['faceId'])
    #print(len(data[0]))

    conn.close()
except Exception as e:
    print("[Errno {0}] {1}".format(e.errno, e.strerror))

for i in faceId:
    headers_1 = {
        # Request headers
        'Content-Type': 'application/json',
        'Ocp-Apim-Subscription-Key': '486688345a164773bf6fb69d6466fd04',
    }
    params_1 = urllib.parse.urlencode({
    })
    body_1 = {
    		'personGroupId' : 'innovlabv1',
            'faceIds':[
                '%s'%i
            ],
            'maxNumOfCandidatesReturned' : 10,
            'confidenceThreshold' : 0.2
    }
    try:
        conn = http.client.HTTPSConnection('westcentralus.api.cognitive.microsoft.com')
        conn.request("POST", "/face/v1.0/identify?%s" % params_1, str(body_1), headers_1)
        response = conn.getresponse()
        data = response.read()
        print(data)
        conn.close()
    except Exception as e:
        print("[Errno {0}] {1}".format(e.errno, e.strerror))

plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.show()

##Junxiao a244dda4-f793-464b-ac85-5b76c19717f9
##Joseph 3190f43e-4b85-4e4f-8306-2455bbc1246d
##Jay f0d2cf95-ca9b-464a-ac83-dc00f2d307a1
##Triveni 89c84981-d9ed-4d86-85df-246996e77450
##Melanie 887b6907-22f6-4459-b5fc-64343368eaf2
##Tonghui be8ab25f-a92e-42ce-8330-c651048d3df8
#Yishuo b478e91d-fffa-4527-a7ce-125a5b285fc