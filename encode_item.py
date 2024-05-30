import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

column_names = ["movie_id", "title", "release_date", "video_release_date", "IMDb_URL", "unknown", "Action", "Adventure", "Animation", 
                "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", 
                "Romance", "Sci-Fi", "Thriller", "War", "Western"]
movie_data = pd.read_csv('dataset/u.item', sep='|', header=None, names=column_names, encoding='latin-1')

def extract_genres(row):
    genres = row[6:]
    selected_genres = genres[genres == 1].index.tolist()
    return " Genre: " + ', '.join(selected_genres)

movie_data['genres'] = movie_data.apply(extract_genres, axis=1)
movie_data['text'] = "Movie title: " + movie_data['title'] + " " + movie_data['genres']
print(movie_data.head())

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').cuda()

def encode_text(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length').to('cuda')
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().detach()

tqdm.pandas()
embeddings = torch.stack(movie_data['text'].progress_apply(encode_text).tolist(), dim=0).numpy()
np.save('movie_embeddings.npy', embeddings)

pca = PCA(n_components=64)
reduced_embeddings = pca.fit_transform(embeddings)

np.save('movie_reduced_embeddings.npy', reduced_embeddings)

