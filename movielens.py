import pandas as pd
import dgl
import os
import torch

class MovieLens(object):
    def __init__(self, directory):
        '''
        directory: path to movielens directory which should have the three
                   files:
                   users.dat
                   movies.dat
                   ratings.dat
        '''
        self.directory = directory

        users = []
        movies = []
        ratings = []

        # read users
        with open(os.path.join(directory, 'users.dat')) as f:
            for l in f:
                id_, gender, age, occupation, zip_ = l.split('::')
                users.append({
                    'id': int(id_),
                    'gender': gender,
                    'age': age,
                    'occupation': occupation,
                    'zip': zip_,
                    })
        self.users = pd.DataFrame(users).set_index('id')

        # read movies
        with open(os.path.join(directory, 'movies.dat'), encoding='latin1') as f:
            for l in f:
                id_, title, genres = l.split('::')
                genres_set = set(genres.split('|'))
                data = {'id': int(id_), 'title': title}
                for g in genres_set:
                    data[g] = 1
                movies.append(data)

        self.movies = pd.DataFrame(movies).set_index('id')
        # print(self.movies)

        # read ratings
        with open(os.path.join(directory, 'ratings.dat')) as f:
            for l in f:
                user_id, movie_id, rating, timestamp = [int(_) for _ in l.split('::')]
                ratings.append({
                    'user_id': user_id,
                    'movie_id': movie_id,
                    'rating': rating,
                    'timestamp': timestamp,
                    })
        self.ratings = pd.DataFrame(ratings)

        # randomly generate training-validation-test set on the ratings table
        test_set = self.ratings.sample(frac=0.05, random_state=1).index
        valid_set = self.ratings.sample(frac=0.05, random_state=2).index
        valid_set = valid_set.difference(test_set)
        self.ratings['valid'] = self.ratings.index.isin(valid_set)
        self.ratings['test'] = self.ratings.index.isin(test_set)

    def todglgraph(self):
        '''
        returns:
        g, user_ids, movie_ids:
            The DGL graph itself.  Each edge has a binary feature "valid" and a binary
            feature "test" indicating validation/test example.
            The list of user IDs (node i corresponds to user user_ids[i])
            The list of movie IDs (node i + len(user_ids) corresponds to movie movie_ids[i])
        '''
        user_ids = list(self.users.index)
        movie_ids = list(self.movies.index)

        # print(self.movies[0])
        user_ids_invmap = {id_: i for i, id_ in enumerate(user_ids)}
        movie_ids_invmap = {id_: i for i, id_ in enumerate(movie_ids)}

        g = dgl.DGLGraph()
        g.add_nodes(len(user_ids) + len(movie_ids))
        rating_user_vertices = [user_ids_invmap[id_] for id_ in self.ratings['user_id'].values]
        rating_movie_vertices = [movie_ids_invmap[id_] + len(user_ids)
                                 for id_ in self.ratings['movie_id'].values]
        valid_tensor = torch.from_numpy(self.ratings['valid'].values.astype('uint8'))
        test_tensor = torch.from_numpy(self.ratings['test'].values.astype('uint8'))
        g.add_edges(rating_user_vertices,
                    rating_movie_vertices,
                    data={'valid': valid_tensor, 'test': test_tensor})
        g.add_edges(rating_movie_vertices,
                    rating_user_vertices,
                    data={'valid': valid_tensor, 'test': test_tensor})

        return g, user_ids, movie_ids



if __name__ == "__main__":
    directory = "data/ml-1m"
    m = MovieLens(directory)
