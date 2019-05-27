import pandas as pd
import numpy as np


r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('../Downloads/Machine-Learning/DataScience-Python3/ml-100k/u.data', sep='\t', names=r_cols, usecols=range(3), encoding="ISO-8859-1")

m_cols = ['movie_id', 'title']
movies = pd.read_csv('../Downloads/Machine-Learning/DataScience-Python3/ml-100k/u.item', sep='|', names=m_cols, usecols=range(2), encoding="ISO-8859-1")

ratings = pd.merge(movies, ratings)

#print(ratings)

movieRatings = ratings.pivot_table(index=['user_id'],columns=['title'],values='rating')


print(movieRatings.head())



starWarsRatings = movieRatings['Star Wars (1977)']

#print(starWarsRatings.head())

similarmovies=movieRatings.corrwith(starWarsRatings)
#print(similarmovies.head(1))

similarmovies=similarmovies.dropna()
#print(similarmovies.head(100))

df=pd.DataFrame(similarmovies)
#print(df.head)

similarmovies=similarmovies.sort_values(ascending=False)
df=pd.DataFrame(similarmovies)
#print(df.head(20))

moviestats=ratings.groupby('title').agg({'rating':[np.size, np.mean]})
#print(moviestats.head)

popularmovies=moviestats['rating']['size']>=200
#print(moviestats[popularmovies])


moviestats=moviestats[popularmovies].sort_values([('rating','mean')], ascending=False)

#print(moviestats.head(40))

df=moviestats[popularmovies].join(pd.DataFrame(similarmovies, columns=['similarity']))

similarity=df.sort_values(['similarity']ascending=False)


print(similarity.head(25))
