import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def generateRoc(predictList,test_data, confidenceLists):
	fprs = []
	tprs = []
	for thrshld in range(101):
		tp = tn = fp = fn = 0
		thrshld = thrshld / 100
		for i in range(len(predictList)):
			curr_Conf = confidenceLists[i]

			sum_true = sum(j >= thrshld for j in curr_Conf)
			#print(sum_true)
			predict_same = predictList[i] == test_data[i]
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

		print('fn',fn)
		print('tp',tp)
		print('fp',fp)
		print('tn',tn)
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
	print(tprs)
	print(fprs)
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



thrshld = 0.0025

with open('kairos_conf', 'rb') as fp:
	predict_list = pickle.load(fp)
	confidence_lists = pickle.load(fp)
	test_x = pickle.load(fp)

# print(predict_list)
# print(len(predict_list))
# print(test_x)
# print(len(test_x))
cnt = 0
for i in range(len(predict_list)):
	if predict_list[i] == test_x[i]:
		norm = [float(j)/sum(confidence_lists[i]) for j in confidence_lists[i]]
		if thrshld <= max(norm):
			cnt+=1
		print(max(norm))
print(cnt/len(predict_list))
plt.imshow(confusion_matrix(test_x, predict_list))
plt.title('Confusion Matrix for Kairos API at LFW dataset')
plt.show()
#generateRoc(predict_list, test_x, confidence_lists)
