

import numpy as np
from keras.models import Model, load_model
from .duelgame import DuelGame
from fischer.dql import prediction_best, Stats, five_number_summary
from copy import copy
from typing import Union
from math import comb
from trueskill import Rating, rate_1vs1

# from multiprocessing import Pool # TODO



__all__ = ['tournament', 'save_tournament_results_csv', 'load_tournament_results_csv', 'merge_tournament_results', 'get_score_per_model', 'sort_results_by_scores', 'tournament_results_str', 'model_scores_str', 'number_of_pairings']

def number_of_pairings(num_contestants: int) -> int:
    '''
    Return the number of pairings that can exist between any two contestants, where there are `contestants` total contestants.  Order does not matter, so the pairing `(A, B)` is identical to `(B, A)`, and it is not counted twice toward this total.
    '''
    return comb(num_contestants, 2)

def tournament(game_cls: type[DuelGame], model_fpaths: list[str], model_names: list[str], model_ratings: dict[str, Rating], games_per_pairing: int, predict_batch_size: int = 32, workers: int = 4, verbose: bool = False, record_statistics: bool = False) -> Union[dict[tuple[str, str], tuple[float, float]], tuple[dict[tuple[str, str], tuple[float, float]], dict[str, Stats]]]:
    '''
    Simulate a large number of games between all pairs of contestants loaded from `model_fpaths`, and return the results as a `dict` which maps pairings of the model names from `model_names` to the result of play between those two contestants.

    For each pairing, `games_per_pairing` games are individually set up and played between the two models, where every other game has sides swapped in order to eliminate the "side bias" in winning probability.  For this reason, it is better to choose an even number for `games_per_pairing`.

    Total number of games to be played throughout the round-robin tournament is equal to `math.comb(len(model_fpaths), 2) * games_per_pairing`.

    If `record_statistics=True`, then return a tuple of the tournament results and the recorded statistics for each model.  Otherwise, only return the tournament results.

    The dict `model_ratings` is modified in place according to tournament results.
    '''
    if record_statistics:
        stats = {str(model_name): Stats() for model_name in model_names}
    for model_name in model_names:
        if model_name not in model_ratings:
            r = Rating()
            if verbose:
                print(f'No Rating found in model_ratings for model {model_name}.  Assuming default rating ({r}).')
            model_ratings[model_name] = r
    games_per_pairing = max(games_per_pairing, 2)
    _model_names = {fpath: str(name) for fpath, name in zip(model_fpaths, model_names)}
    models: dict[str, Model] = {}
    for fpath, name in zip(model_fpaths, model_names):
        if fpath == 'random' or name == 'random': # TODO Allow randomly playing agent in tournaments
            continue
        model = load_model(fpath, compile=False)
        models[fpath] = model
        if isinstance(model.input_shape, list):
            models[fpath]([
                np.zeros((1, *s[1:]))
                for s in model.input_shape
            ])
        else:
            models[fpath](np.zeros((1, *models[fpath].input_shape[1:])))
        print(f'Loaded model {name}')
    pairings = [(fp0, fp1) for i, fp0 in enumerate(model_fpaths[:-1]) for fp1 in model_fpaths[i+1:]]
    total_pairings = len(pairings)
    all_games: list[tuple[int, int, DuelGame]] = [(pairing_index, pairing_game_index, game_cls()) for pairing_index in range(total_pairings) for pairing_game_index in range(games_per_pairing)]
    print(f'Total games to play in tournament: {len(all_games)}')
    print(f'Number of games per contestant: {games_per_pairing * (len(model_fpaths) - 1)}')
    finished = False
    do_reset = True
    while not finished:
        games_per_model: dict[str, list[DuelGame]] = {fpath: [] for fpath in model_fpaths}
        for pi, pgi, g in all_games:
            if do_reset:
                g._reset()
            if g.is_terminal():
                continue
            fp0, fp1 = pairings[pi]
            if g.get_turn() == (pgi % 2 == 0):
                games_per_model[fp0].append(g)
            else:
                games_per_model[fp1].append(g)
            if g.is_terminal():
                print('ERROR!')
        do_reset = False
        fp_to_play = max(model_fpaths, key=lambda fp: len(games_per_model[fp]))
        model_to_play = models.get(fp_to_play, 'random')
        if verbose:
            print(f'Stepping for model {_model_names[fp_to_play]} on {len(games_per_model[fp_to_play])} games')
        masks = np.empty((len(games_per_model[fp_to_play]), *model_to_play.output_shape[1:]), dtype=bool)
        if isinstance(model_to_play.input_shape, list):
            states = [
                np.empty((len(games_per_model[fp_to_play]), *s[1:]), dtype=float)
                for s in model_to_play.input_shape
            ]
            for i, g in enumerate(games_per_model[fp_to_play]):
                for j, s in enumerate(g.get_state()):
                    states[j][i] = s
                masks[i] = g.get_action_mask()
        else:
            states = np.empty((len(games_per_model[fp_to_play]), *model_to_play.input_shape[1:]), dtype=float)
            for i, g in enumerate(games_per_model[fp_to_play]):
                states[i] = g.get_state()
                masks[i] = g.get_action_mask()
        if isinstance(states, list) and states[0].shape[0] <= 16 * predict_batch_size or states.shape[0] <= 16 * predict_batch_size:
            predictions = model_to_play(states).numpy()
        else:
            predictions = model_to_play.predict(states, batch_size=predict_batch_size, verbose=False)
        if record_statistics:
            stats[fp_to_play].record('test_q_values', five_number_summary(predictions), max_length=None)
        rewards = np.empty(len(states))
        # predictions = model_to_play.predict_on_batch(states)
        actions = prediction_best(predictions, vectorized=True, mask=masks)
        # TODO: Use multiprocessing to simulate games faster
        for i, (g, action) in enumerate(zip(games_per_model[fp_to_play], actions)):
            rewards[i] = -g.get_static_reward()
            neg = not g.get_turn()
            g._step(action)
            rewards[i] += g.get_static_reward()
            if neg:
                rewards[i] *= -1
        if record_statistics:
            stats[fp_to_play].record('test_reward_per_step', np.mean(rewards))
        finished = True
        for _, _, g in all_games:
            if not g.is_terminal():
                finished = False
        del masks, states, predictions, actions
    scores = {(_model_names[fp0], _model_names[fp1]): (0., 0.) for fp0, fp1 in pairings}
    for pi, pgi, g in all_games:
        fp0, fp1 = pairings[pi]
        names = _model_names[fp0], _model_names[fp1]
        result = g._get_result()
        if result == (pgi % 2 == 1):
            scores[names] = scores[names][0] + 1, scores[names][1]
            r0, r1 = rate_1vs1(model_ratings[names[0]], model_ratings[names[1]], drawn=False)
            model_ratings[names[0]] = r0
            model_ratings[names[1]] = r1
        elif result == (pgi % 2 == 0):
            scores[names] = scores[names][0], scores[names][1] + 1
            r0, r1 = rate_1vs1(model_ratings[names[1]], model_ratings[names[0]], drawn=False)
            model_ratings[names[1]] = r0
            model_ratings[names[0]] = r1
        else: # result == -1
            scores[names] = scores[names][0] + .5, scores[names][1] + .5
            r0, r1 = rate_1vs1(model_ratings[names[0]], model_ratings[names[1]], drawn=True)
            model_ratings[names[0]] = r0
            model_ratings[names[1]] = r1
    del model_fpaths, _model_names, models
    if record_statistics:
        return scores, stats
    return scores

def save_tournament_results_csv(fpath: str, results: dict[tuple[str, str], tuple[float, float]]):
    '''
    Save tournament results to a CSV file.
    '''
    arr = [['model_name_0', 'model_name_1', 'score_0', 'score_1']]
    for (name0, name1), (score0, score1) in results.items():
        arr.append((name0, name1, str(score0), str(score1)))
    with open(fpath, 'w') as f:
        f.write(
            '\n'.join(
                ','.join(row)
                for row in arr
            )
        )

def load_tournament_results_csv(fpath: str) -> dict[tuple[str, str], tuple[float, float]]:
    '''
    Load a tournament results from a CSV file.
    '''
    results = {}
    with open(fpath, 'r') as f:
        line_iter = iter(f)
        next(line_iter) # skip the first line (the header)
        for line in line_iter:
            line = line.strip()
            split = line.split(',')
            # print(split)
            model0, model1, score0, score1 = split
            score0 = float(score0)
            score1 = float(score1)
            results[(model0, model1)] = score0, score1
    return results

def merge_tournament_results(results0: dict[tuple[str, str], tuple[float, float]], results1: dict[tuple[str, str], tuple[float, float]]) -> dict[tuple[str, str], tuple[float, float]]:
    '''
    Merge the results from one tournament with the results of another, returning a new `dict`.

    The returned results contains pairings that follow the order of those in `results0`.
    For example, if model pairing `(A, B)` in `results0` exists as `(B, A)` in `results1`, then the returned results contains the pairing `(A, B)` and not `(B, A)`.
    '''
    new_results = copy(results0)
    for (model0, model1), (score0, score1) in results1.items():
        if (model1, model0) in new_results:
            score0, score1 = score1, score0
            names = model1, model0
        else:
            names = model0, model1
        if names not in new_results:
            new_results[names] = score0, score1
        else:
            new_results[names] = new_results[names][0] + score0, new_results[names][1] + score1
    return new_results

def get_score_per_model(tournament_results: dict[tuple[str, str], tuple[float, float]]) -> dict[str, float]:
    '''
    Return a `dict` that maps model names to their overall tournament scores.
    '''
    scores = {}
    for (name0, name1), (score0, score1) in tournament_results.items():
        if name0 not in scores:
            scores[name0] = score0
        else:
            scores[name0] += score0
        if name1 not in scores:
            scores[name1] = score1
        else:
            scores[name1] += score1
    return scores

def sort_results_by_scores(tournament_results: dict[tuple[str, str], tuple[float, float]]) -> list[tuple[str, float]]:
    '''
    Similar to `get_score_per_model()`, but instead a `list` is returned with tuples that pair each model in `results` with its overall score.

    The list is sorted in descending order by model scores.
    '''
    scores = get_score_per_model(tournament_results)
    return list(sorted(
        scores.items(),
        key=lambda t: t[1],
        reverse=True,
    ))

def tournament_results_str(results: dict[tuple[str, str], tuple[float, float]]) -> str:
    '''
    Nicely format the results of a tournament as a string to be printed.
    '''
    arr = [('Model 1', 'Model 2', 'Result')]
    for (name0, name1), (score0, score1) in results.items():
        arr.append((name0, name1, f'{score0} - {score1}'))
    l = [
        max(len(row[i]) for row in arr)
        for i in range(3)
    ]
    s = ''
    for i, row in enumerate(arr):
        if i > 0:
            s += '\n'
        for j in range(3):
            s += '|'
            s += ' ' + row[j] + ' ' * (l[j] - len(row[j]) + 1)
        s += ' ' * (len(row[2]) - l[2]) + '|'
    return s

def model_scores_str(scores: Union[dict[str, float], list[tuple[str, float]]]):
    '''
    Nicely format a `list` or `dict` of scores for each model as a string to be printed.

    These lists and dictionaries may be generated from tournament results via `sort_results_by_scores()` or `get_score_per_model()` respectively.
    '''
    if isinstance(scores, dict):
        scores = list(sorted(
            scores.items(),
            key=lambda t: t[1],
            reverse=True,
        ))
    s = ''
    for i, (name, score) in enumerate(scores):
        if i > 0:
            s += '\n'
        s += f'{name}: {score}'
    return s
