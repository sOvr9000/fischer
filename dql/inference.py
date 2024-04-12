

import numpy as np
from typing import Union
from fischer.math.common import sigmoid



__all__ = ['prediction_best', 'prediction_best_multi_action', 'prediction_weighted_choice', 'prediction_weighted_choice_multi_action', 'random_action', 'random_action_multi_action', 'five_number_summary']

def prediction_best(pred: np.ndarray, vectorized: bool = True, mask: np.ndarray[bool] = None) -> Union[int, tuple, np.ndarray]:
    '''
    Out of the Q-values that correspond to `True` in `mask`, return the index of the highest Q-value.

    If `mask = None`, then assume `mask` is `True` for each Q-value.  Otherwise, `mask` must have the same shape as `pred` (even if `vectorized = True`).
    '''
    if mask is None:
        mask = np.ones(pred.shape, dtype=bool)
    assert mask.shape == pred.shape
    if pred.dtype is not np.float_:
        pred = pred.astype(float)
    else:
        pred = pred.copy()
    pred[np.logical_not(mask)] = -np.inf
    if len(pred.shape) == 1:
        return np.argmax(pred) # int
    if len(pred.shape) == 2 and vectorized:
        return np.argmax(pred, axis=1) # np.ndarray
    if vectorized:
        f = np.reshape(pred, (pred.shape[0], -1))
        am = np.argmax(f, axis=1)
        i = np.unravel_index(am, pred.shape[1:]) # tuple of np.ndarray
        c = np.row_stack(i)
        return c # np.ndarray
    f = np.reshape(pred, (-1,))
    am = np.argmax(f)
    i = np.unravel_index(am, pred.shape)
    return i # tuple

def prediction_best_multi_action(pred: list[np.ndarray], possible_actions: Union[list[tuple], list[list[tuple]]], vectorized: bool = True) -> Union[tuple, np.ndarray]:
    if vectorized:
        arr = np.empty((pred[0].shape[0], len(possible_actions[0][0])), dtype=int)
        for i, action_set in enumerate(possible_actions):
            arr[i] = max(
                action_set,
                key=lambda t: sum(pred[g][i, a] for g, a in enumerate(t))
            )
        return arr
    return max(
        possible_actions,
        key=lambda t: sum(pred[g][a] for g, a in enumerate(t))
    )

def prediction_weighted_choice(pred: np.ndarray, vectorized: bool = True, bias: float = 0, mask: np.ndarray[bool] = None) -> Union[int, tuple, np.ndarray]:
    '''
    Weighted choice based on predicted Q-values.

    If `bias > 0`, then higher Q-values are weighted more heavily.
    If `bias < 0`, then the weights are more evenly distributed, so there's less bias in choosing actions with better Q-values.

    `mask` functions identically as in `prediction_best()`.
    '''
    if mask is None:
        mask = np.ones(pred.shape, dtype=bool)
    assert mask.shape == pred.shape
    pred = pred.copy()
    def safe_divide(p: np.ndarray[float], s: float) -> np.ndarray[float]:
        if s == 0:
            total_size = 1
            for l in p.shape:
                total_size *= l
            return np.ones(p.shape) / total_size
        p /= s
        return p
    def safe_divide_vec(p: np.ndarray[float], s: np.ndarray[float]) -> np.ndarray[float]:
        total_size = 1
        for l in p.shape[1:]:
            total_size *= l
        s0 = s == 0
        return np.where(
            s0,
            np.ones(p.shape) / total_size,
            p / np.where(
                s0,
                1,
                s
            )
        )
    def mask_p(p: np.ndarray[float]) -> np.ndarray:
        p[np.logical_not(mask)] = 0
        if vectorized:
            s = np.sum(p, axis=1, keepdims=True)
            return safe_divide_vec(p, s)
        else:
            s = np.sum(p)
            return safe_divide(p, s)
    if len(pred.shape) == 1:
        d = np.sqrt(np.sum(np.square(pred)))
        n = safe_divide(pred, d)
        p = np.exp((15 * sigmoid(bias)) * n)
        s = p.sum()
        p = mask_p(safe_divide(p, s))
        return np.random.choice(pred.shape[0], p=p) # int
    if len(pred.shape) == 2 and vectorized:
        d = np.sqrt(np.sum(np.square(pred), axis=1, keepdims=True))
        n = safe_divide_vec(pred, d)
        p = np.exp((15 * sigmoid(bias)) * n)
        p = mask_p(safe_divide_vec(p, np.sum(p, axis=1, keepdims=True)))
        c = np.empty(pred.shape[0], dtype=int)
        l = pred.shape[1]
        for k, prob in enumerate(p):
            if np.any(np.isinf(prob)):
                c[k] = np.random.randint(l)
            else:
                c[k] = np.random.choice(l, p=prob)
        return c # np.ndarray
    if vectorized:
        f = np.reshape(pred, (pred.shape[0], -1))
        fm = np.reshape(mask, (pred.shape[0], -1))
        c = prediction_weighted_choice(f, vectorized=True, bias=bias, mask=fm)
        i = np.unravel_index(c, pred.shape[1:]) # tuple of np.ndarray
        r = np.column_stack(i)
        return r # np.ndarray
    f = np.reshape(pred, (-1,))
    fm = np.reshape(mask, (-1,))
    c = prediction_weighted_choice(f, vectorized=False, bias=bias, mask=fm)
    i = np.unravel_index(c, pred.shape) # tuple of np.ndarray
    return i # tuple

def prediction_weighted_choice_multi_action(pred: list[np.ndarray], possible_actions: Union[list[tuple], list[list[tuple]]], vectorized: bool = True, bias: float = 0) -> Union[tuple, np.ndarray[int]]:
    '''
    Weighted choice based on predicted Q-values, where each prediction of a state is a sequence of 1D action spaces that comprise the overall N-dimensional action space.

    If `bias > 0`, then higher Q-values are weighted more heavily.
    If `bias < 0`, then the weights are more evenly distributed, so there's less bias in choosing actions with better Q-values.
    '''
    if vectorized:
        actions = np.empty((pred[0].shape[0], len(possible_actions[0][0])))
        for k, (_pred, _possible_actions) in enumerate(zip(zip(*pred), possible_actions)):
            actions[k] = prediction_weighted_choice_multi_action(_pred, _possible_actions, vectorized=False, bias=bias)
        return actions
    if len(possible_actions) == 0:
        raise Exception(f'Cannot choose a random action when none of the actions are possible.')
    p = np.empty(len(possible_actions), dtype=float)
    for i, action in enumerate(possible_actions):
        p[i] = sum(pred[g][a] for g, a in enumerate(action))
    p /= p.sum()
    p = np.exp((15 * sigmoid(bias)) * p)
    p /= p.sum()
    return tuple(possible_actions[np.random.choice(np.arange(len(p)), p=p)])

def random_action(action_shape: Union[int, tuple, np.ndarray], vectorized: bool = True, mask: np.ndarray[bool] = None):
    if mask is None:
        mask = np.ones(action_shape, dtype=bool)
    assert mask.shape == action_shape, f'Shapes not equivalent, received shapes: mask={mask.shape}, action={action_shape}'
    if isinstance(action_shape, int):
        return prediction_weighted_choice(np.ones(action_shape) / action_shape, vectorized=vectorized, mask=mask)
    def array_shape_total_length(shape: tuple) -> int:
        if len(shape) == 1:
            return shape[0]
        return array_shape_total_length(shape[1:]) * shape[0]
    return prediction_weighted_choice(np.ones(action_shape) / array_shape_total_length(action_shape), vectorized=vectorized, mask=mask)

def random_action_multi_action(possible_actions: Union[list[int], list[list[int]]], vectorized: bool = True) -> Union[tuple, np.ndarray[int]]:
    if vectorized:
        actions = np.empty((len(possible_actions), len(possible_actions[0][0])))
        for k, action_set in enumerate(possible_actions):
            actions[k] = random_action_multi_action(action_set, vectorized=False)
        return actions
    return possible_actions[np.random.randint(len(possible_actions))]

def five_number_summary(pred: Union[list[np.ndarray], np.ndarray]) -> tuple[float, float, float, float, float]:
    '''
    Return the average min, average 25th percentile, mean, average 75th percentile, and average max of the Q-values in `pred`.

    `pred` is assumed to be shaped such that axis `0` represents the batch axis, so the returned average min (for example) is the average of the minimal Q-value from each prediction in `pred`.
    '''
    if isinstance(pred, list):
        return five_number_summary(np.concatenate(tuple(np.reshape(p, (p.shape[0], -1)) for p in pred), axis=1))
    f = np.reshape(pred, (pred.shape[0], -1))
    return f.min(axis=1).mean(), np.percentile(f, 25, axis=1).mean(), f.mean(), np.percentile(f, 75, axis=1).mean(), f.max(axis=1).mean()


