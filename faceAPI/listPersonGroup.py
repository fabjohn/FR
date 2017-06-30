import http.client, urllib.request, urllib.parse, urllib.error, base64

headers = {
    # Request headers
    'Ocp-Apim-Subscription-Key': '486688345a164773bf6fb69d6466fd04',
}

params = urllib.parse.urlencode({
    # Request parameters
    #'start': '{string}',
    'top': '1000',
})
body = {}
try:
    conn = http.client.HTTPSConnection('westcentralus.api.cognitive.microsoft.com')
    conn.request("GET", "/face/v1.0/persongroups?%s" % params, str(body), headers)
    response = conn.getresponse()
    data = response.read()
    print(data)
    conn.close()
except Exception as e:
    print("[Errno {0}] {1}".format(e.errno, e.strerror))