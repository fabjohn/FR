import boto3

BUCKET = 'amazon-rekognition'
KEY = 'porsche.jpg'

rekognition = boto3.client('rekognition','us-west-2')
with open(KEY, 'rb') as dock_image:
    dock_bytes = dock_image.read()

response = rekognition.detect_labels(
    Image = { 'Bytes' : dock_bytes
    },
    MaxLabels = 10,
    MinConfidence = 50,
)
print(response['Labels'])