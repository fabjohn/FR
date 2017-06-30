##compare faces

import boto3

BUCKET = 'amazon-rekognition'
KEY_1 = 'test6.jpg'
KEY_2 = 'test3.jpg'


rekognition = boto3.client('rekognition','us-west-2')
with open(KEY_1, 'rb') as face_image:
    face_bytes = face_image.read()
with open(KEY_2, 'rb') as test_image:
    test_bytes = test_image.read()

response = rekognition.compare_faces(
	SourceImage = {'Bytes' : face_bytes},
	TargetImage = {'Bytes' : test_bytes},
	SimilarityThreshold = 30,
)

print('Source Face ({Confidence}%)'.format(**response['SourceImageFace']))
for match in response['FaceMatches']:
	print('target face ({Confidence}%)'.format(**match['Face']))
	print ('  Similarity : {}%'.format(match['Similarity']))