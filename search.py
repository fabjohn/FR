import boto3

BUCKET = 'amazon-rekognition'
KEY = 'jo1.jpg'
COLLECTION = 'test'

rekognition = boto3.client('rekognition','us-west-2')
with open(KEY, 'rb') as tst_image:
	tst_bytes = tst_image.read()
response = rekognition.search_faces_by_image(
	Image = { 'Bytes' : tst_bytes
	},

	CollectionId = COLLECTION,
	FaceMatchThreshold = 50,

)

x = response['FaceMatches']
for face in x:
	temp = face['Face']
	print(temp['FaceId'])
	print(face['Similarity'])
