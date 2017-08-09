#!/usr/bin/python

import sys
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import SVC,OneClassSVM
from sklearn.metrics import precision_recall_fscore_support
from puLearning.puAdapter import PUAdapter
from sklearn import metrics
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.cluster import DBSCAN, Birch, AgglomerativeClustering

def onehot_encode(data, const_len=64):
    """ expand n to a vector v with length 256, v[n] = 1, v[other] = 0 """
    res = []
    for n in data:
        ex = [0] * 257
        ex[n] = 1
        res += ex
    length=len(data)
    while length<const_len :
	res.extend([0]*256)
	res.append(1)
	length+=1
	  #  print len(temp)
    return res

def read_sample_from_file(f, number = -1 ):
    res = []
    
    if number < 0:
        while True:
            length = f.read(1)
            if not length:
                break
            s = f.read(ord(length))
            res.append([ord(i) for i in s])
    else:
        n = 0
        while True:
            length =f.read(1)
            if not length:
                break
            length=ord(length)
            s = f.read(length)
            res.append([ord(i) for i in s])
	    
            n += 1
            if n >= number:
                break
    return res

def OneClass_estimator(X, y):
    # random rank the data
    permut = np.random.permutation(len(y))
    X = X[permut]
    y = y[permut]
    print "Splitting dataset"
    print
    split=2*len(y)/3
    X_train=X[:split]
    y_train = y[:split]
    X_test = X[split:]
    y_test = y[split:]

    pu_f1_scores = []
    reg_f1_scores = []

    pos = np.where(y_train == +1.)[0]
    np.random.shuffle(pos)
    rng = np.random.RandomState(42)
    estimator=IsolationForest(max_samples=200,bootstrap=True, random_state=rng)
    #estimator=OneClassSVM(nu=0.1, kernel="rbf", gamma=0.5)
    print len(X_train[pos])
    estimator.fit(X_train[pos])
    y_pred=estimator.predict(X_test)
    print len(np.where(y_train == -1.)[0])," are negtive"
    print len(np.where(y_train == +1.)[0])," are positive"
    print
    print y_pred
    print len(np.where(y_pred == 1.)[0])
    print len(np.where(y_test == 1.)[0])
  
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)
    pu_f1_scores.append(f1_score[1])
    print "F1 score: ", f1_score[1]
    print "Precision: ", precision[1]
    print "Recall: ", recall[1]
    print

def PU_estimater(X, y):
    # random rank the data
    permut = np.random.permutation(len(y))
    X = X[permut]
    y = y[permut]
    print "Splitting dataset"
    print
    split=2*len(y)/3
    X_train=X[:split]
    y_train = y[:split]
    X_test = X[split:]
    y_test = y[split:]

    pu_f1_scores = []
    reg_f1_scores = []

    n_sacrifice_iter = range(100, 700, 200 )
    for n_sacrifice in n_sacrifice_iter:
	print "Making ", n_sacrifice, " positive examples negtive."
        print
        y_train_pu = np.copy(y_train)
        pos = np.where(y_train == +1.)[0]
        np.random.shuffle(pos)
        sacrifice = pos[:n_sacrifice]
        y_train_pu[sacrifice] = -1.
        print "PU transformation applied. We now have:"

        print len(np.where(y_train_pu == -1.)[0])," are negtive"
        print len(np.where(y_train_pu == +1.)[0])," are positive"
        print
        
        #Get f1 score with pu_learning
        print "PU learning in progress..."
        estimator = RandomForestClassifier(n_estimators=100,
                                           criterion='gini', 
                                           bootstrap=True,
                                           n_jobs=1)
       # estimator = SVC(C=10,
        #            kernel='rbf',
         #           gamma=0.4,
          #          probability=True)
   
	pu_estimator = PUAdapter(estimator,hold_out_ratio=0.2)
        pu_estimator.fit(X_train,y_train_pu)
	y_pred = pu_estimator.predict(X_test)
	
        precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)
        pu_f1_scores.append(f1_score[1])
        print "F1 score: ", f1_score[1]
        print "Precision: ", precision[1]
        print "Recall: ", recall[1]
        print
	sac_test=X_train[sacrifice] 
        sar_pred=pu_estimator.predict(sac_test)
	ratio=float(len(np.where(sar_pred==+1.)[0]))/n_sacrifice
	print "sacrifice positive samples are labeled as:",sar_pred
	print "ratio of the correct labeled sacrifice pos_samples:",ratio
	print
       
	estimator = RandomForestClassifier(n_estimators=100,
                                           criterion='gini', 
                                           bootstrap=True,
                                           n_jobs=1)
	estimator.fit(X_train, y_train_pu)
	y_pred_re=estimator.predict(X_test)
        precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred_re)
        pu_f1_scores.append(f1_score[1])
        print "F1 score: ", f1_score[1]
        print "Precision: ", precision[1]
        print "Recall: ", recall[1]
        print
	sac_test=X_train[sacrifice] 
        sar_pred=estimator.predict(sac_test)
	ratio=float(len(np.where(sar_pred==+1.)[0]))/n_sacrifice
	print "sacrifice positive samples are labeled as:",sar_pred
	print "ratio of the correct labeled sacrifice pos_samples:",ratio
	print	


def Clustering_map(X, y):
    # random rank the data
    permut = np.random.permutation(len(y))
    X = X[permut]
    y = y[permut]
    print "Splitting dataset"
    print
    split=2*len(y)/3
    X_train=X[:split]
    y_train = y[:split]
    X_test = X[split:]
    y_test = y[split:]

    pu_f1_scores = []
    reg_f1_scores = []

    n_sacrifice_iter = range(500, 600, 100 )
    for n_sacrifice in n_sacrifice_iter:
	print "number of the labeled data ", n_sacrifice, " to map clusters."
        print
        pos = np.where(y_train == +1.)[0]
        np.random.shuffle(pos)
        sacrifice = pos[:n_sacrifice]
        print "PU transformation applied. We now have:"

        print len(np.where(y_train == -1.)[0])," are negtive"
        print len(np.where(y_train == +1.)[0])," are positive"
        print

	#cluster_estimator=DBSCAN(eps=0.3, min_samples=20)

	#core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
	#core_samples_mask[db.core_sample_indices_] = True
	#labels = db.labels_

	#n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
	n_clusters=30
	cluster_estimator=MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, batch_size=45, n_init=10, max_no_improvement=10, verbose=0)
#	cluster_estimator=AgglomerativeClustering(n_clusters=n_clusters,
#                                    linkage="average", affinity="euclidean")
	
	cluster_estimator.fit(X_train)
	cluster_estimator_means_cluster_centers = np.sort(cluster_estimator.cluster_centers_, axis=0)

	average_occur=float(n_sacrifice)/len(X_train)
	print average_occur
	sac_data=X_train[sacrifice] 
	cluster_estimator_means_labels=pairwise_distances_argmin(sac_data, cluster_estimator_means_cluster_centers)
	cluster_estimator_means_labels=np.array(cluster_estimator_means_labels)

	train_labels=pairwise_distances_argmin(X_train, cluster_estimator_means_cluster_centers)
	train_labels=np.array(train_labels)
	ratio=[]
	for i in range(0, n_clusters):
		sub_divided= len(np.where(train_labels == i)[0])
		if sub_divided == 0:
			temp=0
		else:
			temp=float(len(np.where(cluster_estimator_means_labels == i)[0]))/len(np.where(train_labels == i)[0])
		#print len(np.where(cluster_estimator_means_labels == i)[0])
		#print len(np.where(train_labels == i)[0])
		#print temp
		ratio.append(temp) 
	ratio=np.array(ratio)
	cluster_index_map=np.where(ratio > average_occur)[0]

	test_labels=pairwise_distances_argmin(X_test, cluster_estimator_means_cluster_centers)
	test_labels=np.array(test_labels)
	
	map_test_labels=[]
	for i in test_labels:
		if i in cluster_index_map:
			map_test_labels.append(1)
		else:
			map_test_labels.append(-1)
#	print ratio
#	print map_test_labels
#	print "cluster of labeled data:", cluster_estimator_means_labels
        #print "homo score:", metrics.homogeneity_score(y_test, map_test_labels)
        #print "comple score:", metrics.completeness_score(y_test, map_test_labels)
        #print "v score:", metrics.v_measure_score(y_test, map_test_labels)
	

        precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, map_test_labels)
        pu_f1_scores.append(f1_score[1])
        print "F1 score: ", f1_score[1]
        print "Precision: ", precision[1]
        print "Recall: ", recall[1]
        print

def main():

    if len(sys.argv) < 3:
        print "Usage: %s <pos> <neg>" % sys.argv[0]
        sys.exit(0)
    # set the feature vectors 
    pos_samples = read_sample_from_file(open(sys.argv[1]),10000)
    neg_samples = read_sample_from_file(open(sys.argv[2]),20000)
    print "%d positive samples found!" % len(pos_samples)
    print "%d negetive samples found!" % len(neg_samples)
    samples=pos_samples+neg_samples
    X=[]
    X+=[onehot_encode(i) for i in samples]
    X=np.array(X)

    # set the labels
    pos_labels=np.ones(len(pos_samples),dtype=np.int64)
   # print np.where(pos_labels==1.)[0]
    neg_labels=-1*np.ones(len(neg_samples),dtype=np.int64) 
   # print np.where(neg_labels==-1.)[0]
    labels=np.append(pos_labels,neg_labels)
   # print np.where(labels==1)[0] 
    y=labels 
   # PU_estimater(X,y)
    #Clustering_map(X,y)
    OneClass_estimator(X,y)    

main()

