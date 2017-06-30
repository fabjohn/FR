import boto3

client = boto3.client('rekognition','us-west-2')
# response = client.list_faces(
# 	CollectionId ='test',
# )
# faces = response['Faces']
# print(faces[0])

# response = client.delete_collection(
# 	CollectionId='test'
# )

# print(response)

response = client.list_collections(
)
print(response)