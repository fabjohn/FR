from time import time
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import fetch_lfw_people

people = fetch_lfw_people('./faces', min_faces_per_person=70,resize=0.4)

print(people.shape())
