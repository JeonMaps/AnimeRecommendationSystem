

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

anime = pd.read_csv('anime.csv')
anime.head(10)
anime.info()
anime.describe()

anime_dup = anime[anime.duplicated()]
print(anime_dup)
type_values = anime['type'].value_counts()
print(type_values)

m = anime['members'].quantile(0.75)
print(m)

qualified_anime = anime.copy().loc[anime['members'] > m]
C = anime['rating'].mean()


def WR(x, C=C, m=m):
    v = x['members']
    R = x['rating']
    return (v/(v+m)*R)+(m/(v+m)*C)

qualified_anime['score'] = WR(qualified_anime)
qualified_anime.sort_values('score', ascending=False)
qualified_anime.head(15)


tf_idf = TfidfVectorizer(lowercase=True, stop_words='english')
anime['genre'] = anime['genre'].fillna('')
tf_idf_matrix = tf_idf.fit_transform(anime['genre'])
tf_idf_matrix.shape

cosine_sim = linear_kernel(tf_idf_matrix, tf_idf_matrix)
indices = pd.Series(anime.index, index=anime['name'])
indices = indices.drop_duplicates()

def recommendations(name, cosine_sim=cosine_sim):
    similarity_scores = list(enumerate(cosine_sim[indices[name]]))
    similarity_scores = sorted(
        similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:21]
    anime_indices = [i[0] for i in similarity_scores]
    return anime['name'].iloc[anime_indices]

recommendations('Kimi no Na wa.')


count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(anime['genre'])
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
recommendations('Kimi no Na wa.', cosine_sim2)


cosine_sim2 = linear_kernel(count_matrix, count_matrix)
recommendations('Kimi no Na wa.', cosine_sim2)
