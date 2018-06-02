######################################################################
# This program is implementation of three algorithms - Baseline Model,
# user-user collaborative filtering, item-item collaborative filtering.
#
# The file takes the input as the path of kaggle_triplets_eval_new
# Commandline : python collaborative_filtering.py filepath
# 
# Output : Global mean , RMSE for baseline algorith, RMSE for User-user
# algorithm and RMSE for item-item algorithm 
#
# @Author: Richa Jain
# @Date: June 1, 2018
# 
#####################################################################

#importing the libraries
import numpy as np
import pandas as pd
from sklearn import cross_validation as cv
from sklearn.metrics.pairwise import pairwise_distances,cosine_similarity
from sklearn.metrics import mean_squared_error, precision_score,accuracy_score
from math import sqrt
import matplotlib.pyplot as plt
import sys

#Creates the list of song_no for each song which is used in generation of user-item profile
def song_no(song_id):
	sid={}
	j=0
	for i in song_id:
		if i not in sid:
			sid.update({i:j})
			j=j+1
	return sid

#Creates the list of user_no for each song which is used in generation of user-item profile	
def user_no(user_id):
	uid ={}
	j=0
	for i in user_id:
		if i not in uid:
			uid.update({i:j})
			j=j+1
	return uid

#Creates the train matrix, of user-item with playcount as values
def training_matrix(train_data_matrix):
	for line in train_data.itertuples():
		train_data_matrix[line[6], line[5]] = line[4]
	return train_data_matrix

#Creates the test matrix, of user-item with playcount as values
def testing_matrix(test_data_matrix):
	for line in test_data.itertuples():
		test_data_matrix[line[6], line[5]] = line[4]
	return test_data_matrix

#The function predicts the rating matrix based on user-user or item-item similarity
def predict(ratings, similarity, type='user'):
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
        
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred

#The function computes RMSE value 
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth))

#The function calculates the rating based on the baseline model
def baseline_model(baseline_value,test_data_matrix,user_bias,song_bias):
	ir1=0
	ir2=0
	for i in test_data_matrix:
		jr1=0
		for j in i:
			if j==0:
				user_bias[ir1][jr1]=0-baseline_value
			else:
				ub1 = np.mean(i)
                #ub1=sum(i)/np.count_nonzero(i)
				#ub1=np.mean(i)
				user_bias[ir1][jr1]=ub1-baseline_value
			jr1=jr1+1
		ir1=ir1+1
	for i in test_data_matrix.T:
		jr2=0
		for j in i:
			if j==0:
				song_bias[ir2][jr2]=0-baseline_value
			else:
				#sb1=sum(i)/np.count_nonzero(i)
				sb1=np.mean(i)
				song_bias[ir2][jr2]=sb1-baseline_value
			jr2=jr2+1
		ir2=ir2+1 
	return song_bias.T + user_bias + baseline_value


if __name__=='__main__':
	user_path = sys.argv[1] #takes the command line input
	df = pd.read_csv(user_path)
	df=df[df['playcount']<=3] # filters the input to scale of 1 to 3
	baseline_value =np.mean(df['playcount']) # finds the global mean of playcount
	print('Global Average:', baseline_value)
	song_id=df.songid
	user_id=df.userid
	df['songno'] = df['songid'].map(song_no(song_id))
	df['userno'] =df['userid'].map(user_no(user_id))
	n_users = df.userno.unique().shape[0]
	n_songs =df.songno.unique().shape[0]
	print('Number of unique users = ' + str(n_users) + ' | Number of unique songs = ' + str(n_songs))
	train_data, test_data = cv.train_test_split(df, test_size=0.25) # splits the data intp train and test set
	
	#creates train matrix, user-user and item-item similarity matrix
	train_data_matrix = np.zeros((n_users, n_songs))
	trian_data_matrix = training_matrix(train_data_matrix)
	user_similarity = pairwise_distances(train_data_matrix, metric='cosine')
	item_similarity = pairwise_distances(train_data_matrix.T, metric='cosine')
	
	#calculcates the prediction based on both item-item and user-user similarity
	item_prediction = predict(train_data_matrix, item_similarity, type='item')
	user_prediction = predict(train_data_matrix, user_similarity, type='user')
	test_data_matrix = np.zeros((n_users, n_songs))
	test_data_matrix = testing_matrix(test_data_matrix)
	print('User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
	print('Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix)))
	song_bias=np.zeros((n_users,n_songs)).T
	user_bias=np.zeros((n_users,n_songs))
	baseline_res = baseline_model(baseline_value,test_data_matrix,user_bias,song_bias) #computes the baseline prediction
	print('Baseline Model with Bias RMSE:'+str(rmse(baseline_res,test_data_matrix)))
