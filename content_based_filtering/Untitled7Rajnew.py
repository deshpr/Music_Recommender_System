
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
df_album = pd.read_csv('C:\\Users\\rnedunur\\Documents\\Python Scripts\\IR\\kaggle_songs.csv')
df_album.head()
df_album.drop(df_album.columns[[0, 1]], axis=1, inplace=True)
df_album.head()
df_tracks = pd.read_csv('C:\\Users\\rnedunur\\Documents\\Python Scripts\\IR\\unique_tracks.csv',encoding = "ISO-8859-1")
df_tracks.head()
df_tracks.drop(df_tracks.columns[[0]], axis=1, inplace=True)
df_tracks.head()
df_artist = pd.read_csv('C:\\Users\\rnedunur\\Documents\\Python Scripts\\IR\\unique_artists.csv',encoding = "ISO-8859-1")
df_artist.drop(df_artist.columns[[0]], axis=1, inplace=True)
df_artist.head()
df_playcnt = pd.read_csv('C:\\Users\\rnedunur\\Documents\\Python Scripts\\IR\\kaggle_visible_evaluation_triplets.txt', sep="\t", header=None)
df_playcnt.columns = ["userid", "songid", "playcount"]
df_playcnt.head()
catalog_prep=pd.merge(pd.merge(df_album, df_tracks, on='songid'),df_artist, on='artistname')
catalog_set= pd.merge(catalog_prep,df_playcnt, on='songid')
df_artist_filtered = pd.read_csv('C:\\Users\\rnedunur\\Documents\\Python Scripts\\IR\\Filtered_Artists.csv',encoding = "ISO-8859-1", header=None)
df_artist_filtered.columns = ["artistname"]
df_artist_filtered.head()
final_df= pd.merge(catalog_set,df_artist_filtered, on ='artistname')
final_df.head()
final_df.to_csv('C:\\Users\\rnedunur\\Documents\\Python Scripts\\IR\\Processed\\final_df.csv')
item_matrix_prep=final_df.groupby(['songid','artistname'],as_index=False)
item_matrix_prep=item_matrix_prep.aggregate(np.sum)
item_matrix = pd.get_dummies(item_matrix_prep[['songid','artistname']], columns=['artistname'],prefix=[None])
item_matrix.shape
item_matrix.to_csv('C:\\Users\\rnedunur\\Documents\\Python Scripts\\IR\\Processed\\item_matrix.csv')
user_matrix_prep=final_df.groupby(['userid','artistname'],as_index=False)
user_matrix_prep=user_matrix_prep.aggregate(np.sum)
user_matrix = pd.pivot_table(user_matrix_prep, values='playcount', index='userid', columns=['artistname'], aggfunc=np.sum,fill_value=0)
user_matrix.shape
user_matrix.to_csv('C:\\Users\\rnedunur\\Documents\\Python Scripts\\IR\\Processed\\user_matrix.csv')
user_matrix.describe()
user_normalized= (user_matrix - user_matrix.values.min()) / (user_matrix.values.max() - user_matrix.values.min())
user_normalized.describe()
user_normalized.to_csv('C:\\Users\\rnedunur\\Documents\\Python Scripts\\IR\\Processed\\user_normalized.csv')


# In[80]:


item_mat_cpy= item_matrix.copy(deep=True)
item_mat_cpy['cosine'] =0.0
item_matrix_same_shape = item_matrix.drop(item_matrix.columns[[0]], axis=1)


# In[89]:


user_song_mat=final_df[['userid','songid']]
user_song_mat['val'] =1
user_song_mat=pd.pivot_table(user_song_mat, 'val', 'userid', 'songid', aggfunc='sum', fill_value=0)


# In[ ]:


df_output = pd.DataFrame(index= range(1),columns=['userid','songid','cosine'])
df_output = df_output.fillna(0) # with 0s rather than NaNs

from scipy.spatial.distance import cosine
for i in range(0,len(user_normalized.index)-1):
    for j in range(0,len(item_matrix_same_shape.index)-1):
         if(user_song_mat.ix[i,j]== 0):
            item_mat_cpy.at[j, 'cosine'] = 1 - cosine(item_matrix_same_shape.iloc[j], user_normalized.iloc[i])
    df_tmp=item_mat_cpy[['songid','cosine']].nlargest(10,'cosine')
    df_tmp['userid'] = user_normalized.index[i]
    df_output=df_output.append(df_tmp)
    item_mat_cpy['cosine'] =0.0
