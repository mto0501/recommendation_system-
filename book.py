from PIL.Image import new
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

menu = ["Home","About","Data Set"]

#choice = st.sidebar.selectbox("Menu",menu)
#if choice == "Home":
@st.cache()
def load_data(db_path):
    df = pd.read_csv(db_path) #,sep='\t'
    return df


def recommend(df, title, category):
  # match cate with column cat in DataFrame
  data = df.loc[df["categories"]== category] ###
  data.reset_index(level=0,inplace=True)
  #print('data:', data.shape)
  

  indices = pd.Series(data.index,index=data["title"])
  print('indices', indices)
  #TFIDF bi,tri-gram
  tf = TfidfVectorizer(min_df=2,    
                       max_df=0.7, 
                       ngram_range=(2,3), 
                       analyzer='word',
                       stop_words='english')
  tfidf_matrix = tf.fit_transform(data['new_des'])

  # Calculate the similarity 
  similarity = cosine_similarity(tfidf_matrix,tfidf_matrix)
  print(similarity.shape)

  #Get index of original title
  index = indices[title]
  print(type(index))

  if isinstance(index,np.int64) == False:
      index=index[0]

  print('index=', index)

  #similarity score
  similarity = list(enumerate(similarity[index]))
  print(similarity)

  #sort books
  similarity = sorted(similarity, key=lambda x: x[1], reverse=True)

  # GET TOP 5 MOST SIMILAR BOOKS
  similarity  = similarity [1:6]

  book_indices = [i[0] for i in similarity]

  #top 5 recommendation 
  rec = data[['title', 'thumbnail']].iloc[book_indices]

  # print title 
  return rec['title']

  # print top 5 books cover
#   for i in rec['thumbnail']:
#         response = requests.get(i)
#         img = Image.open(BytesIO(response.content))
#         plt.figure()
#         plt.imshow(img)


df = load_data('df3_new.csv')
#df= pd.read_csv("df3_new.csv")
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Home":
    st.title('Programming Book Recommendation System')
    col1, col2 = st.columns(2)
    with col1:
        input_title = st.text_input('Enter Title : ')

    with col2:
        input_genre = st.text_input('Enter Gerne : ')
        
    st.write("Key Search:")
    st.write("Harry Potter and the Half-Blood Prince (Harry Potter  #6), Juvenile Fiction")
    #Voyager (Outlander #3)
    st.write("Whores for Gloria, Fiction")  
    #Harry Potter and the Chamber of Secrets (Harry Potter #2)
    if st.button('Search'):
        st.success('Recommending books similar to '+ input_title)
        print('input:', input_title, input_genre)
        new_title = recommend(df, input_title, input_genre)
        for book in new_title:
            st.write(book)
    



if choice == "Data Set":
     df1 = pd.read_csv("df3_new.csv")
     st.dataframe(df1)

     

