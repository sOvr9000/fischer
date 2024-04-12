
import os
import numpy as np
from keras.models import Model, load_model
from .memory import TransitionMemory
from .models import create_target_model
from typing import Union
# from memory_profiler import profile




def experience_replay(models: Union[tuple[Model, Model], Model], num_transitions: int, memory: TransitionMemory, fit_epochs: int = 1, predict_batch_size: int = 6128, fit_batch_size: int = 32, gamma: float = 0.98, tau: float = 0.2, verbose: bool = False, save_target_model: bool = False):
    '''
    Perform experience replay with a model and its target model (double DQL).

    `models` is of the form `(online_model, target_model)`.  Alternatively, `models` can be an instance of a `keras.Model`, in which case a target model is automatically created and is automatically used for the `models` model on successive calls.
    '''

    if isinstance(models, Model):
        online_model = models
        if not hasattr(experience_replay, 'associated_target_models'):
            experience_replay.associated_target_models: dict[tuple, Model] = {}
        key = tuple(models.input_shape)
        s = str(hash(f'{online_model.input_shape} - {online_model.output_shape}')) + '.h5'
        if not os.path.isdir('./target_models'):
            os.makedirs('./target_models')
        if key in experience_replay.associated_target_models:
            target_model = experience_replay.associated_target_models[key]
        elif os.path.isfile('./target_models/' + s):
            target_model = load_model('./target_models/' + s, compile=False)
            target_model.compile(online_model.optimizer, online_model.loss)
            experience_replay.associated_target_models[key] = target_model
        else:
            target_model = create_target_model(models)
            experience_replay.associated_target_models[key] = target_model
        models = online_model, target_model
    else:
        online_model, target_model = models

    if memory.is_multi_action:
        _experience_replay_multi_actions(models=models, num_transitions=num_transitions, memory=memory, fit_epochs=fit_epochs, predict_batch_size=predict_batch_size, fit_batch_size=fit_batch_size, gamma=gamma, tau=tau, verbose=verbose, save_target_model=save_target_model)
        return

    from_states, to_states, rewards, actions, terminal, next_valid_actions = memory.sample(num_transitions)

    if len(actions.shape) > 1:
        # these are multi-index actions, which could be from either a single >1-dimensional output or multiple 1-dimensional outputs
        multi_index_actions = True
        actions = np.ravel_multi_index(actions.T, online_model.output_shape[1:])
    else:
        multi_index_actions = False

    if isinstance(from_states, list):
        pred_from = online_model([
            np.concatenate((fs, ts), axis=0)
            for fs, ts in zip(from_states, to_states)
        ]).numpy()
        pred_to = pred_from[from_states[0].shape[0]:]
        pred_from = pred_from[:from_states[0].shape[0]]
    else:
        pred_from = online_model(np.concatenate((from_states, to_states), axis=0)).numpy()
        pred_to = pred_from[from_states.shape[0]:]
        pred_from = pred_from[:from_states.shape[0]]

    pred_to_target = target_model(to_states).numpy()
    pred_to[np.logical_not(next_valid_actions)] = -np.inf

    if multi_index_actions:
        original_shape = pred_from.shape
        pred_from = pred_from.reshape((pred_from.shape[0], -1))
        pred_to = pred_to.reshape((pred_to.shape[0], -1))
        pred_to_target = pred_to_target.reshape((pred_to_target.shape[0], -1))

    arange = np.arange(num_transitions, dtype=int)
    pred_from[arange, actions] = rewards + gamma * np.logical_not(terminal) * pred_to_target[arange, np.argmax(pred_to, axis=1)]

    if multi_index_actions:
        pred_from = pred_from.reshape(original_shape)

    online_model.fit(from_states, pred_from, batch_size=fit_batch_size, epochs=fit_epochs, shuffle=fit_epochs>1, verbose=verbose)
    # shuffle=False because memory sampling is random, or shuffle=True when the indices are reused (epochs > 1)

    w = online_model.get_weights()
    tw = target_model.get_weights()
    target_model.set_weights([_w * tau + _tw * (1 - tau) for _w, _tw in zip(w, tw)])

    if save_target_model:
        if not os.path.isdir('./target_models'):
            os.makedirs('./target_models')
        target_model.save('./target_models/' + str(hash(f'{online_model.input_shape} - {online_model.output_shape}')) + '.h5')
    
    del online_model, target_model, from_states, to_states, rewards, actions, terminal, next_valid_actions, multi_index_actions, pred_from, pred_to, pred_to_target, original_shape, arange, w, tw



def _experience_replay_multi_actions(models: Union[tuple[Model, Model], Model], num_transitions: int, memory: TransitionMemory, fit_epochs: int = 1, predict_batch_size: int = 6128, fit_batch_size: int = 32, gamma: float = 0.98, tau: float = 0.2, verbose: bool = False, save_target_model: bool = False):
    online_model, target_model = models
    from_states, to_states, rewards, actions, terminal, next_valid_actions = memory.sample(num_transitions)
    is_multi_state = isinstance(online_model.input_shape, list)

    if is_multi_state:
        pred_from = [
            outp.numpy()
            for outp in online_model([
                np.concatenate((fs, ts), axis=0)
                for fs, ts in zip(from_states, to_states)
            ])
        ]
        pred_to = [
            pf[from_states[0].shape[0]:]
            for pf in pred_from
        ]
        for i in range(len(pred_from)):
            pred_from[i] = pred_from[i][:from_states[0].shape[0]]
    else:
        pred_from = [
            outp.numpy()
            for outp in online_model(np.concatenate((from_states, to_states), axis=0))
        ]
        pred_to = [
            pf[from_states.shape[0]:]
            for pf in pred_from
        ]
        for i in range(len(pred_from)):
            pred_from[i] = pred_from[i][:from_states.shape[0]]

    pred_to_target = [
        outp.numpy()
        for outp in target_model(to_states)
    ]

    max_q_actions = np.empty((len(next_valid_actions), len(next_valid_actions[0])), dtype=int)
    for i, action_set in enumerate(next_valid_actions):
        max_q_actions[i] = max(action_set, key=lambda t: sum(pred_to[g][i][a] for g, a in enumerate(t)))

    pred_to_target_mean_q = np.empty(len(next_valid_actions))
    for i, action in enumerate(max_q_actions):
        pred_to_target_mean_q[i] = sum(pred_to_target[g][i][a] for g, a in enumerate(action)) / max_q_actions.shape[1]

    new_q = rewards + gamma * np.logical_not(terminal) * pred_to_target_mean_q
    arange = np.arange(num_transitions, dtype=int)
    for g, _actions in enumerate(max_q_actions.T):
        pred_from[g][arange, _actions] = new_q

    online_model.fit(from_states, pred_from, batch_size=fit_batch_size, epochs=fit_epochs, shuffle=fit_epochs>1, verbose=verbose)
    # shuffle=False because memory sampling is random, or shuffle=True when the indices are reused (epochs > 1)

    w = online_model.get_weights()
    tw = target_model.get_weights()
    target_model.set_weights([_w * tau + _tw * (1 - tau) for _w, _tw in zip(w, tw)])

    if save_target_model:
        if not os.path.isdir('./target_models'):
            os.makedirs('./target_models')
        target_model.save('./target_models/' + str(hash(f'{online_model.input_shape} - {online_model.output_shape}')) + '.h5')
    
    del online_model, target_model, from_states, to_states, rewards, actions, terminal, next_valid_actions, multi_index_actions, pred_from, pred_to, pred_to_target, pred_to_target_mean_q, max_q_actions, original_shape, arange, w, tw


