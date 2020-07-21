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
        for i in range(1,6):
            recommendations.append(get_title_from_index(sorted_similar_movies[i][0]))
        moviesDB = imdb.IMDb()
        id1 = moviesDB.search_movie(recommendations[0])[0].getID()
        id2 = moviesDB.search_movie(recommendations[1])[0].getID()
        id3 = moviesDB.search_movie(recommendations[2])[0].getID()
        id4 = moviesDB.search_movie(recommendations[3])[0].getID()
        id5 = moviesDB.search_movie(recommendations[4])[0].getID()
        # IDs = []
        # for i in range(0,5):
        #     IDs.append(moviesDB.search_movie(recommendations[i])[0].getID())
        list_of_movies = []
        list_of_movies.append(moviesDB.get_movie(id1))
        list_of_movies.append(moviesDB.get_movie(id2))
        list_of_movies.append(moviesDB.get_movie(id3))
        list_of_movies.append(moviesDB.get_movie(id4))
        list_of_movies.append(moviesDB.get_movie(id5))
        # for i in range(0,5):
        #     list_of_movies.append(moviesDB.get_movie(IDs[i]))

        year_of_movies = []
        cover_images = []
        title_of_movie = []
        rating_of_movie = []
        for i in range(0,5):
            year_of_movies.append(list_of_movies[i]['year'])
            cover_images.append(list_of_movies[i]['cover url'])
            title_of_movie.append(list_of_movies[i]['title'])
            rating_of_movie.append(list_of_movies[i]['rating'])

        print(year_of_movies)
        print(title_of_movie)
        print(rating_of_movie)
        print(cover_images)
        return render_template('result.html', year_of_movies = year_of_movies,cover_images = cover_images, recommendations=recommendations,title_of_movie = title_of_movie,rating_of_movie = rating_of_movie)
    
    except :
        print('Exception occured')
        
if __name__ == "__main__":
    app.run()