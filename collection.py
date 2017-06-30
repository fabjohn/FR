##Create collection

import boto3

BUCKEET = 'amazon-rekognition'

rekognition = boto3.client('rekognition','us-west-2')
response = rekognition.create_collection(CollectionId = 'test',)
print(response)