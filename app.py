from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import imdb

app = Flask(__name__, static_folder='templates/static')



@app.route('/', methods=['GET', 'POST'])
def form():
    return render_template('index.html')

@app.route('/hello', methods=['GET', 'POST'])
def hello():  
    try:
        df = pd.read_csv("movie_dataset.csv")

        def get_title_from_index(index):
            return df[df.index == index]["title"].values[0]

        def get_index_from_title(title):
            return df[df.title == title]["index"].values[0]



        features = ['keywords','cast','genres','director']
        for feature in features:
            df[feature] = df[feature].fillna('')
        def combine_features(row):
            return row['keywords']+" "+row["cast"]+" "+row["genres"]+" "+row['director']
        df["combined_features"] = df.apply(combine_features,axis=1)
        cv = CountVectorizer()
        count_matrix = cv.fit_transform(df["combined_features"])
        cosine_sim = cosine_similarity(count_matrix)
        myMovie = request.form['myMovie']
        movie_index = get_index_from_title(myMovie)
        similar_movies = list(enumerate(cosine_sim[movie_index]))
        sorted_similar_movies = sorted(similar_movies,key = lambda x:x[1],reverse=True)
        recommendations = []
        for i in range(1,11):
            recommendations.append(get_title_from_index(sorted_similar_movies[i][0]))
        moviesDB = imdb.IMDb()
        movies = moviesDB.search_movie(myMovie)
        id = movies[0].getID()
        movie = moviesDB.get_movie(id)
        title = movie['title']
        year=movie['year']
        rating=movie['rating']
        # directors=movie['directors']
        casting=movie['cast']
        print(myMovie)
        return render_template('result.html', myMovie=myMovie, recommendations=recommendations,title=title,year=year,rating=rating,casting=casting)
    
    except :
        print('Exception occured')
        
if __name__ == "__main__":
    app.run()
