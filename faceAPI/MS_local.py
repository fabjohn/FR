import matplotlib.pyplot as plt
import cv2 
import time
import numpy as np
from sklearn.datasets import fetch_lfw_people
import os
import sys
import pickle
from sklearn.metrics import confusion_matrix



def generateRoc(predictList,test_data, confidenceLists, nameId):
	fprs = []
	tprs = []
	for thrshld in range(101):
		tp = tn = fp = fn = 0
		thrshld = thrshld / 100
		for i in range(len(predictList)):
			curr_Conf = confidenceLists[i]

			sum_true = sum(j >= thrshld for j in curr_Conf)
			#print(sum_true)
			predict_same = predictList[i] == nameId[test_data[i]]
			if sum_true>0 and predict_same:
				tp += 1
				fp += sum_true - 1
				tn += len(curr_Conf) - sum_true
			elif sum_true>0 and not predict_same:
				fp += sum_true
				tn += len(curr_Conf) - sum_true
			elif sum_true<= 0 and predict_same:
				fn += 1
				tn += len(curr_Conf) - 1
			elif sum_true<= 0 and not predict_same:
				tn += len(curr_Conf)


		if tp + fn ==0:
			tpr = 0
		else:
			tpr = float(tp) / float(tp+fn)
		if fp + tn == 0:
			fpr = 0
		else:
			fpr = float(fp) / float(fp+tn)
		fprs.append(fpr)
		tprs.append(tpr)

	AUC = getAUC(fprs, tprs)
	print(AUC)
	plt.plot(fprs, tprs)
	plt.xlabel('fpr')
	plt.ylabel('tpr')
	plt.title('ROC curve')
	plt.show()


def getAUC(fprs, tprs):
    sortedFprs, sortedTprs = zip(*sorted(zip(*(fprs, tprs))))
    sortedFprs = list(sortedFprs)
    sortedTprs = list(sortedTprs)
    if sortedFprs[-1] != 1.0:
        sortedFprs.append(1.0)
        sortedTprs.append(sortedTprs[-1])
    return np.trapz(sortedTprs, sortedFprs)



with open('accuracy_file1', 'rb') as fp:
	nameId = pickle.load(fp)
	test_x = pickle.load(fp)
with open('accuracy_file2', 'rb') as fp:
	predictList = pickle.load(fp)
	confidenceLists = pickle.load(fp)
	del_list = pickle.load(fp)
for index in sorted(del_list, reverse=True):
	del test_x[index]
confidenceLists.sort()
print(confidenceLists)
# if len(predictList) != len(test_x):
# 	print('this is not right. fuck')

cnt = 0
temp = []
for i in range(len(predictList)):
	if predictList[i] == nameId[test_x[i]]:
 		cnt +=1
	temp.append(nameId[test_x[i]])
accuracy = cnt/len(predictList)
print(accuracy)
plt.imshow(confusion_matrix(temp, predictList))
plt.title('Confusion Matrix for Microsoft Face API at LFW dataset')
plt.show()
#0.9003667481662592 for 1636 people 


# # print(len(predictList))
# # print(predictList)
# #print(confidenceLists)
# ##genereate ROC curve
# generateRoc(predictList, test_x, confidenceLists, nameId)



