import pickle
import re
import pandas as pd
import numpy as np
import random
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def GetRecommendations():
    def importDataset():
        with open("static/dataset/headlinesFile.data", "rb") as filehandle:
          headlines = pickle.load(filehandle)

        with open("static/dataset/corpusFile.data", "rb") as filehandle:
          content = pickle.load(filehandle)

        with open("static/dataset/dateFile.data", "rb") as filehandle:
          date_of_creation = pickle.load(filehandle)

        return headlines, content, date_of_creation

    def preprocess_text(text):
      sen=re.sub('["/n","/t"]',"",text) #removing newlines
      sen=re.sub('[^0-9a-zA-Z]', ' ', text) #removing numbers and puncutations
      sen=re.sub(r"\s+[a-zA-Z]\s+", ' ',text) #removing single characters
      sen=re.sub(r"\s+"," ",text) #removing multiple spaces
      return sen

    def get_docs(id):
        return (id,dataset[dataset.id==id].Content.values[0],dataset[dataset.id==id].Headlines.values[0])

    def create_dummy_users():
      l=[]
      user_id=[]                    #Use drop_dulicates
      doc_id=[]
      click_frequency=[]
      for i in range(1500):
        t=(random.randint(0,500),random.randint(0,1138))
        if not t in l:
          user_id.append(t[0])
          doc_id.append(t[1])
          l.append(t)
          click_frequency.append(random.randint(0,5))
      user_data=pd.DataFrame({"user_id":user_id,"id":doc_id,"click_frequency":click_frequency})
      return user_data

    def create_user_profile():
       id=[]
       avg_vector=[]
       for i in user_data.loc[:,"user_id"]:
         if not i in id:
            sum=0
            id.append(i)

            for j,k in enumerate(user_data[user_data.user_id==i].id):
                  sum+=np.array(tfidf_vector[tfidf_vector.index==k])

            avg_vector.append(np.squeeze(sum/(j+1)))
       df=pd.DataFrame(avg_vector,columns=vectorizer.vocabulary_.keys())
       df["user_id"]=id
       df.set_index("user_id",inplace=True)
       return df

    def get_recommendations(id):
      l=[]
      b=user_doc_similarity[user_doc_similarity.index==id]
      for i,j in zip(b.values[0],b.columns):
        l.append((i,j))
      return sorted(l,reverse=True,key=lambda X:X[0])

    def get_users(id):
      l=[]
      user=profile[profile.index==id].values
      sum=0
      for i,j in   zip(similar_users_table[similar_users_table.index==id].values[0],similar_users_table.columns)  :
        l.append((i,j))
      for i in sorted(l,key=lambda X:X[0],reverse=True)[1:11]:
        sum+=profile[profile.index==i[1]].values
      sum+=user
      return  sum/(10)

    def get_collaborative_recommendations(id):
      l=[]
      user=profile[profile.index==id].values
      sum=0
      for i,j in   zip(similar_users_table[similar_users_table.index==id].values[0],similar_users_table.columns)  :
        l.append((i,j))
      for i in sorted(l,key=lambda X:X[0],reverse=True)[1:11]:
        sum+=profile[profile.index==i[1]].values
      sum+=user
      avg_vec=sum/10

      l1=[]

      for i,j in zip(cosine_similarity(avg_vec,tfidf_vector)[0],tfidf_vector.index):

        l1.append((i,j))
      docs1=[]
      for i in sorted(l1,reverse=True,key=lambda X:X[0]):
        docs1.append(get_docs(i[1]))
      return docs1[:20]

    headlines, content, date_of_creation = importDataset()

    regex=re.compile(r'["\t","\n","\r",",","."]')

    datetime=[]
    for i in range(len(date_of_creation)):
      datetime.append(regex.sub("",date_of_creation[i]))


    #Preprocessing the text

    preprocessed_content=[]
    for i in content:
      preprocessed_content.append(preprocess_text(i))

    preprocessed_headline=[]
    for i in headlines:
      preprocessed_headline.append(preprocess_text(i))

    dataset=pd.DataFrame({"id":[i for i in range(len(headlines))],"Headlines":preprocessed_headline,"Content":preprocessed_content,"date":datetime})

    #Dummy User data creation
    # user_data=create_dummy_users()
    #
    # with open("static/dataset/userFile.data", "wb") as filehandle:
    #   pickle.dump(user_data, filehandle)

    with open("static/dataset/userFile.data", "rb") as filehandle:
      user_data = pickle.load(filehandle)


    combined_data=user_data.merge(dataset,on="id")


    #Vectorize content and headlines
    # nltk.download("stopwords")
    my_stopwords = stopwords.words('english')

    vectorizer = TfidfVectorizer(stop_words = my_stopwords)
    bag_of_words = vectorizer.fit_transform(preprocessed_content)

    tfidf_vector=pd.DataFrame(bag_of_words.todense(),columns=vectorizer.vocabulary_.keys())
    tfidf_vector["doc_id"]=[i for i in range(1154)]
    tfidf_vector.set_index("doc_id",inplace=True)


    #User Profile
    profile=create_user_profile()

    similar_users=cosine_similarity(profile,dense_output=True)
    similar_users_table=pd.DataFrame(similar_users,columns=profile.index,index=profile.index)


    #Content Based Filtering
    user_doc_similarity=pd.DataFrame(cosine_similarity(profile,tfidf_vector),columns=tfidf_vector.index,index=profile.index)

    recommended_docs,docs=[],[]
    for i in get_recommendations(50):
      recommended_docs.append(i[1])
    for i in recommended_docs:
      docs.append(get_docs(i))


    #Collaborative Filtering
    return get_collaborative_recommendations(200)

with open("static/dataset/userFile.data", "rb") as filehandle:
  users = pickle.load(filehandle)

print(users)
