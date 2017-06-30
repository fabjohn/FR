import http.client, urllib.request, urllib.parse, urllib.error, base64
import requests, json

headers = {
	'Content-Type': 'application/json',
    'Ocp-Apim-Subscription-Key': '486688345a164773bf6fb69d6466fd04',
}
name = 'lfwtest'
body = {
	"name":"%s"%name,
    #"userData":"user-provided data attached to the person group"
}
params = urllib.parse.urlencode({
})

try: 
	#r = requests.post('https://westcentralus.api.cognitive.microsoft.com/face/v1.0/persongroups/innovlabv1', json = body, headers = headers)
	conn = http.client.HTTPSConnection('westcentralus.api.cognitive.microsoft.com')
	conn.request('PUT', '/face/v1.0/persongroups/%s?%s' % (name,params), str(body), headers)
	response = conn.getresponse()
	data = response.read()
	print(data)
	conn.close()
	#print(r.json())
except Exception as e:
	print('[Errno {0}] {1}'.format(e.errno, e.strerror))