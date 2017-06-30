import boto3

BUCKET = 'amazon-rekognition'
KEY = 'test.jpg'

rekognition = boto3.client('rekognition','us-west-2')

with open(KEY,'rb') as face_image:
	face_bytes = face_image.read()

response = rekognition.detect_faces(
	Image = { 
		'Bytes' : face_bytes
	},
	Attributes = [
		'ALL'
	],
)

#print(response['FaceDetails'])
for face in response['FaceDetails']:
	print('Face ({Confidence}%)'.format(**face))
	for emotion in face['Emotions']:
		print('{Type} : {Confidence}%'.format(**emotion))