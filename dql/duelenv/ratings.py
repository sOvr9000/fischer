
import os
from trueskill import Rating
import pandas as pd



__all__ = ['load_model_ratings', 'save_model_ratings']

def load_model_ratings(models_dir: str) -> dict[str, Rating]:
    fpath = f'{models_dir}/ratings.csv'
    if not os.path.exists(fpath):
        return {}
    d = pd.read_csv(fpath)
    ratings = {}
    for model_name, mu, sigma in zip(d['model_name'], d['mu'], d['sigma']):
        ratings[model_name] = Rating(mu=mu, sigma=sigma)
    return ratings

def save_model_ratings(models_dir: str, ratings: dict[str, Rating]):
    model_names = []
    mus = []
    sigmas = []
    for model_name, rating in ratings.items():
        model_names.append(model_name)
        mus.append(rating.mu)
        sigmas.append(rating.sigma)
    d = pd.DataFrame({
        'model_name': model_names,
        'mu': mus,
        'sigma': sigmas,
    })
    fpath = f'{models_dir}/ratings.csv'
    d.to_csv(fpath)
