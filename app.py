from flask import Flask, redirect, url_for, render_template, request, session
from datetime import timedelta

import pickle
import re
import pandas as pd
import numpy as np
import random
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = "ThisIsNotAlabama"
app.permanent_session_lifetime = timedelta(days=5)

# with open("static/dataset/users_new.data", "wb") as filehandle:
#     users = {"admin": [1234, 0, 500]}
#     pickle.dump(users, filehandle)

def GetRecommendations(user):
    def importDataset():
        with app.open_resource("static/dataset/headlinesFile.data", "rb") as filehandle:
          headlines = pickle.load(filehandle)

        with app.open_resource("static/dataset/corpusFile.data", "rb") as filehandle:
          content = pickle.load(filehandle)

        with app.open_resource("static/dataset/dateFile.data", "rb") as filehandle:
          date_of_creation = pickle.load(filehandle)

        return headlines, content, date_of_creation

    def preprocess_text(text):
      sen=re.sub('["/n","/t"]',"",text) #removing newlines
      sen=re.sub('[^0-9a-zA-Z]', ' ', text) #removing numbers and puncutations
      sen=re.sub(r"\s+[a-zA-Z]\s+", ' ',text) #removing single characters
      sen=re.sub(r"\s+"," ",text) #removing multiple spaces
      return sen

    def get_docs(id):   #added just now
        return (dataset[dataset.id==id].Content.values[0],dataset[dataset.id==id].Headlines.values[0], id)

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
    def clean_dataset(df):
        assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
        df.dropna(inplace=True)
        indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
        return df[indices_to_keep].astype(np.float64)

    with app.open_resource("static/dataset/userFile.data", "rb") as filehandle:
      user_data = pickle.load(filehandle)

    user_data = clean_dataset(user_data)

    # user_data = user_data.reset_index()
    #user_data = create_dummy_users();

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
    return get_collaborative_recommendations(user)

#Main Webpage
@app.route("/", methods=["POST", "GET"])
def index():
    if "user" in session: #Checks if user is logged in
        if request.method == "POST":
            data = request.form["data"]

            #Open user clickthrough file
            with app.open_resource("static/dataset/userFile.data", "rb") as filehandle:
              userProfile = pickle.load(filehandle)

            with app.open_resource("static/dataset/users_new.data", "rb") as filehandle:
              users = pickle.load(filehandle)

            #TODO: Increase click frequency
            # if users[session["user"][0]][2] in userProfile.user_id and data in userProfile
            # print(89 in userProfile.user_id and 648 in userProfile.id)

            if session["user"][1] == 0:
                users[session["user"][0]][1] = 1
                session["user"][1] = 1
                with open("static/dataset/users_new.data", "wb") as filehandle:
                    pickle.dump(users, filehandle)

            df = pd.DataFrame([[users[session["user"][0]][2], data, 1]], columns=['user_id', 'id', 'click_frequency'])
            userProfile = userProfile.append(df, ignore_index = True)

            with open("static/dataset/userFile.data", "wb") as filehandle:
                pickle.dump(userProfile, filehandle)

            return redirect(url_for("index"))
        else: #GET request
            if session["user"][1] == 0: #When user is new and has no click through data
                with app.open_resource("static/dataset/corpusFile.data", "rb") as filehandle:
                  content = pickle.load(filehandle)
                with app.open_resource("static/dataset/headlinesFile.data", "rb") as filehandle:
                  headlines = pickle.load(filehandle)
                #Creates random numbers from 0 to the len of the articles dataset to give user random articles
                randomNumbers = []
                while len(randomNumbers) < 10:
                    temp = random.randint(0, len(content) - 1)
                    if temp not in randomNumbers:
                        randomNumbers.append(temp)
                randomArticles = []
                for i in randomNumbers:
                    randomArticles.append((content[i], headlines[i], i))
                return render_template("index.html", articles=randomArticles)
            else: #When user has a click through data
                with app.open_resource("static/dataset/users_new.data", "rb") as filehandle:
                  users = pickle.load(filehandle)
                #Session[user] contains {username, new/old} data. Hence, users[session[user][0]]
                #is the same as users[username]. The [2] represents the userid given to the user.
                x = GetRecommendations(users[session["user"][0]][2]);
                return render_template("index.html", articles=x)
    else:
        return redirect(url_for("login"))

#Route for logging in to the web application
@app.route("/login", methods=["POST", "GET"])
def login():
    if "user" in session: #If user is already logged in
        return redirect(url_for("index"))
    else: #If user is not logged in
        if request.method == "POST":
            #Get the username and password entered by the user in the login form
            username = request.form["username"]
            password = request.form["password"]

            #Open the user datafile
            with app.open_resource("static/dataset/users_new.data", "rb") as filehandle:
              users = pickle.load(filehandle)

            if username in users: #If the user is present in the datafile
                if users[username][0] == password: #If the password entered by the user matches with the one in the datafile
                    session.permanent = True
                    #The session data contains [Username, new/old user]
                    session["user"] = [username, users[username][1]]
                    return redirect(url_for("index"))
                else:
                    return render_template("login.html")
            else:
                return render_template("login.html")
        else:
            return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

#Route to make a new user in the database
@app.route("/register", methods=["POST", "GET"])
def register():
    if "user" in session: #If user is already logged in
        return redirect(url_for("index"))
    else: #If user is not logged in
        if request.method == "POST":
            #Get the data entered by the user in the form
            username = request.form["username"]
            password = request.form["password"]

            #Open the user datafile
            with app.open_resource("static/dataset/users_new.data", "rb") as filehandle:
              users = pickle.load(filehandle)

            if username in users: #If user is already registered
                return render_template("login.html")
            else: #Register user in the database
                users[username] = [password, 0, 500 + len(users)] #[password, old/new, userid] 499 is the number of dummy usrs

                #Rewrite the userfile to update it
                with open("static/dataset/users_new.data", "wb") as filehandle:
                    pickle.dump(users, filehandle)
                return render_template("login.html")
        else:
            return render_template("register.html")


if __name__ == "__main__":
    app.run(debug=True)
