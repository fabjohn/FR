import boto3

BUCKET = 'amazon-rekognition'
KEY = 'test4.jpg'
COLLECTION = 'test'
IMAGE_ID = '1'
rekognition = boto3.client('rekognition','us-west-2')
with open(KEY, 'rb') as idx_image:
	idx_bytes = idx_image.read()
response = rekognition.index_faces(
	Image = { 'Bytes' : idx_bytes
	},

	CollectionId = COLLECTION,
	ExternalImageId = IMAGE_ID,
	DetectionAttributes = [],



)
print(response)