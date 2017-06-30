import http.client, urllib.request, urllib.parse, urllib.error, base64
KEY = 'Tonghui/Tonghui_0002.jpg'
headers = {
    # Request headers
    'Content-Type': 'application/octet-stream',
    'Ocp-Apim-Subscription-Key': '486688345a164773bf6fb69d6466fd04',
}
currId = 'edd0a13f-e825-4d5e-ad23-f7ad59a62060'
##Junxiao a244dda4-f793-464b-ac85-5b76c19717f9
##Joseph 3190f43e-4b85-4e4f-8306-2455bbc1246d
##Jay f0d2cf95-ca9b-464a-ac83-dc00f2d307a1
##Triveni 89c84981-d9ed-4d86-85df-246996e77450
##Melanie 887b6907-22f6-4459-b5fc-64343368eaf2
#Tonghui edd0a13f-e825-4d5e-ad23-f7ad59a62060
#Yishuo b478e91d-fffa-4527-a7ce-125a5b285fc
#Kshitij 03090607-d17e-49ae-b00c-0fc6e687134e
params = urllib.parse.urlencode({
    # Request parameters
    'personGroupId' : 'innovlabv1',
    'personId' : '%s' %currId,
    #'userData': '{string}',
    #'targetFace': '{string}',
})
# with open(KEY, 'rb') as test_image:
#     body = test_image.read() 
body = open(KEY,'rb').read()
#print(body)
# body = {
#     'url': 'http://i.imgur.com/Gd0F266.jpg'
# }

try:
    conn = http.client.HTTPSConnection('westcentralus.api.cognitive.microsoft.com')
    conn.request("POST", "/face/v1.0/persongroups/innovlabv1/persons/%s/persistedFaces?%s" % (currId, params), body, headers)
    response = conn.getresponse()
    data = response.read()
    print(data)
    conn.close()
except Exception as e:
    print("[Errno {0}] {1}".format(e.errno, e.strerror))
