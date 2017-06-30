import http.client, urllib.request, urllib.parse, urllib.error, base64

headers = {
    # Request headers
    'Content-Type': 'application/json',
    'Ocp-Apim-Subscription-Key': '486688345a164773bf6fb69d6466fd04',
}

params = urllib.parse.urlencode({
})
body = {'personGroupId' : 'lfwtest',
        'faceIds':[
            '52d2402a-1763-4c84-b7fa-a04ad3d38020'
        ]
}
try:
    conn = http.client.HTTPSConnection('westcentralus.api.cognitive.microsoft.com')
    conn.request("POST", "/face/v1.0/identify?%s" % params, str(body), headers)
    response = conn.getresponse()
    data = response.read()
    print(data)
    conn.close()
except Exception as e:
    print("[Errno {0}] {1}".format(e.errno, e.strerror))