######################################################################
# This program is implementation of Popularity based algorithm for .
# cold start problem.

# The file takes the input as the path of kaggle_triplets_eval_new
# Commandline : python popularity_based.py filepath
# 
# Output : List of top 50 songs for all the users in the test data.
# It also prints the output for 1 user as an example swith top 10  
# songs and top 50 songs for that user.
#
# @Author: Richa Jain
# @Date: June 1, 2018
# 
#####################################################################

#Importing necessary libraries
import pandas as pd
from sklearn import cross_validation as cv
import sys


#Function that creates a list of top songs for the test data users			
def final_recommend(user_list,popularlist):
	songs_recommend={}
	for users in users_list:
		for song in popularlist:
			if users not in songs_recommend:
				songs_recommend.update({users:[song]})
			else:
				songs_recommend[users].append(song)
	return songs_recommend

if __name__=='__main__':

	user_path = sys.argv[1]  #takes the command line input of the file path
	df = pd.read_csv(user_path)
	df = df[df['playcount']<=3] #filters the playcount till 3
	train_data, test_data = cv.train_test_split(df, test_size=0.25) #splits the data into train and test set
	triplets1 = train_data[['songid','playcount']]
	#computes the list of top 50 popular songs
	popularlist=triplets1.groupby(by=['songid']).sum().sort_values(by='playcount',ascending=False).head(50).reset_index()
	popularlist=popularlist['songid'].tolist()
	users_list=test_data['userid'].tolist()
	songs_recommend=final_recommend(users_list,popularlist)
	print('List of Top 10 songs for a single user: ',songs_recommend['08f6fd5e6346af5fb0103b36de72b2c6a87f9314'][0:10])			
	print('List of Top 50 songs for a single user: ',songs_recommend['08f6fd5e6346af5fb0103b36de72b2c6a87f9314'])