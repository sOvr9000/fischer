
import os
import gc
import numpy as np
from typing import Union
from keras.models import Model, clone_model, load_model
from fischer.dql import experience_replay, prediction_best, prediction_best_multi_action, prediction_weighted_choice, TransitionMemory, Stats, five_number_summary, prediction_weighted_choice_multi_action, random_action_multi_action, random_action
from .duelgame import DuelGame
from .ratings import load_model_ratings, save_model_ratings
from fischer.dt import dt
from fischer.progressbar import StylizedProgressBar
from time import sleep
# from memory_profiler import profile
import colorama
from .tournament import *



class DuelEnv:
    '''
    Trains one model to improve against another over many iterations.

    One iteration consists of many one-versus-one games played between the training model and the fixed model.

    Each iteration simulates a set number of games.
    '''
    def __init__(self,
        game_cls: type[DuelGame],
        num_sims: int,
        memory_capacity: int,
        
        fit_epochs: int = 1,
        predict_batch_size: int = 128,
        fit_batch_size: int = 32,
        gamma: float = 0.98,
        tau: float = 0.01,
        exploration_interval: int = 256, # number of steps per exploration-exploitation cycle (beginning of iteration is exploration, gradually approach exploitation, but loop back to exploration cyclically over the duration of the iteration); in general, this should be greater than or equal to the number of steps per episode
        heuristic_interval: int = 384, # like exploration_interval, but this is for when the agent selects an action decided by DuelGame.get_heuristic_action(), which is meant to be a best-guess; heuristic selection overrides random selection
        stats_max_length: int = 16384,
        random_opponent_iterations: int = 3,

        experience_replay_sample_rate: float = 0.75,
        experience_replay_interval: int = 16,
        experience_replays_per_interval: int = 1, # On every experience replay, repeat the experience replay this many times.  A higher number helps with stabilizing the model and more accurately predicting long-term rewards.

        model_save_interval: int = 32,
        model_save_dir: str = None,
    ):

        assert issubclass(game_cls, DuelGame)
        assert isinstance(num_sims, int) and num_sims > 0

        self.game_cls = game_cls
        self.sims: list[DuelGame] = [
            self.game_cls()
            for _ in range(num_sims)
        ]
        for i in range(num_sims):
            self.sims[i].sim_index = i
        self.num_sims = num_sims
        self.statistics = Stats()
        self.memory_capacity = memory_capacity

        self.fit_epochs = fit_epochs
        self.predict_batch_size = predict_batch_size
        self.fit_batch_size = fit_batch_size
        self.gamma = gamma
        self.tau = tau
        self.exploration_interval = exploration_interval
        self.heuristic_interval = heuristic_interval
        self.stats_max_length = stats_max_length
        self.random_opponent_iterations = random_opponent_iterations
        self.step_delay_millis = 0
        self.total_iterations = 0
        self.total_iteration_steps = 0

        self.experience_replay_sample_rate = experience_replay_sample_rate
        self.experience_replay_interval = experience_replay_interval
        self.experience_replays_per_interval = experience_replays_per_interval

        self.model_save_interval = model_save_interval
        if model_save_dir is None:
            model_save_dir = f'./models/{dt()}'
        self.model_save_dir = model_save_dir
        if not os.path.isdir(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        self.model_save_fpath = f'{self.model_save_dir}/current.h5'
        self.data_save_fpath = f'{self.model_save_dir}/data.npz'

        self.fixed_model = None
        self.training_model = None
    def load_training_history_data(self, fpath: str):
        self.statistics.load(fpath)
    def save_training_history_data(self):
        self.statistics.save(self.data_save_fpath)
    def save_training_model(self):
        self.training_model.save(self.model_save_fpath)
    def print_statistics(self):
        print(colorama.Fore.GREEN + 'Q-values\n\t' + colorama.Fore.YELLOW + 'MEAN   {}\n\tSTD    {}\n'.format(self.statistics.mean('train_q_values'), self.statistics.std('train_q_values')))
        print(colorama.Fore.GREEN + 'Reward per episode\n\t' + colorama.Fore.YELLOW + 'MEAN   {:.5f}\n\tSTD    {:.5f}\n'.format(self.statistics.mean('train_reward_per_episode'), self.statistics.std('train_reward_per_episode')))
        print(colorama.Fore.GREEN + 'Reward per step\n\t' + colorama.Fore.YELLOW + 'MEAN   {:.5f}\n\tSTD    {:.5f}\n'.format(self.statistics.mean('train_reward_per_step'), self.statistics.std('train_reward_per_step')))
        print(colorama.Style.RESET_ALL, end='')
    def set_models(self, training_model: Model, fixed_model: Model = None):
        if hasattr(self, 'transition_memory'):
            del self.transition_memory
        if isinstance(training_model.input_shape, list):
            self.transition_memory = TransitionMemory(capacity=self.memory_capacity, state_shape=[s[1:] for s in training_model.input_shape], action_shape=training_model.output_shape[1:])
        else:
            self.transition_memory = TransitionMemory(capacity=self.memory_capacity, state_shape=training_model.input_shape[1:], action_shape=training_model.output_shape[1:])
        self.training_model = training_model
        if fixed_model is None:
            fixed_model = clone_model(self.training_model)
            fixed_model.set_weights(self.training_model.get_weights())
        else:
            _fixed_model = clone_model(fixed_model)
            _fixed_model.set_weights(fixed_model.get_weights())
            fixed_model = _fixed_model
        self.fixed_model = fixed_model
        self.is_multi_action = isinstance(self.training_model.output_shape, list)
    def _get_model_inputs(self) -> Union[np.ndarray[float], list[np.ndarray[float]]]:
        if len(self.training_model.input_shape) > 1 and isinstance(self.training_model.input_shape, list):
            X = [
                np.empty((self.num_sims, *s[1:]))
                for s in self.training_model.input_shape
            ]
            for i, sim in enumerate(self.sims):
                for j, state in enumerate(sim.get_state()):
                    X[j][i] = state
        else:
            X = np.empty((self.num_sims, *self.training_model.input_shape[1:]))
            for i, sim in enumerate(self.sims):
                X[i] = sim.get_state()
        return X
    def _get_action_masks(self) -> np.ndarray[bool]:
        masks = np.empty((self.num_sims, *self.training_model.output_shape[1:]), dtype=bool)
        for i, sim in enumerate(self.sims):
            m = sim.get_action_mask()
            if m is None:
                return None
            masks[i] = m
        return masks
    def _get_valid_actions(self) -> list[list[tuple]]:
        return [
            sim._get_valid_actions()
            for sim in self.sims
        ]
    def _play(self, training_agent: bool):
        if not training_agent and self.is_solo_env(): return
        X = self._get_model_inputs()
        if self.is_multi_action:
            possible_actions = self._get_valid_actions()
        else:
            masks = self._get_action_masks()
        if training_agent:
            if self.is_multi_action:
                if len(X) < self.predict_batch_size:
                    predictions = [
                        outp.numpy()
                        for outp in self.training_model(X)
                    ]
                else:
                    predictions = [
                        outp.numpy()
                        for outp in self.training_model.predict(X, verbose=False, batch_size=self.predict_batch_size)
                    ]
                actions = prediction_weighted_choice_multi_action(predictions, possible_actions, vectorized=True, bias=-10+20*(self.total_iteration_steps % self.exploration_interval / self.exploration_interval))
            else:
                if len(X) < self.predict_batch_size:
                    predictions = self.training_model(X).numpy()
                else:
                    predictions = self.training_model.predict(X, verbose=False, batch_size=self.predict_batch_size)
                actions = prediction_weighted_choice(predictions, vectorized=True, bias=-10+20*(self.total_iteration_steps % self.exploration_interval / self.exploration_interval), mask=masks)
            h_prob = 1 - self.total_iteration_steps % self.heuristic_interval / self.heuristic_interval
            e_prob = 1 - self.total_iteration_steps % self.exploration_interval / self.exploration_interval
            h_select = np.random.random(size=self.num_sims) < h_prob * e_prob
            for i in range(self.num_sims):
                if h_select[i]:
                    a = self.sims[i].get_heuristic_action()
                    if a is None:
                        continue
                    actions[i] = a
            self.statistics.record('train_q_values', five_number_summary(predictions), max_length=self.stats_max_length)
            del h_select
        else:
            if self.total_iterations < self.random_opponent_iterations:
                predictions = None # just set this so that `del predictions` works later
                if self.is_multi_action:
                    actions = random_action_multi_action(possible_actions, vectorized=True)
                else:
                    actions = random_action((masks.shape[0], *self.training_model.output_shape[1:]), vectorized=True, mask=masks)
            else:
                if len(X) < self.predict_batch_size:
                    predictions = self.fixed_model(X).numpy()
                else:
                    predictions = self.fixed_model.predict(X, verbose=False, batch_size=self.predict_batch_size)
                if self.is_multi_action:
                    actions = prediction_best_multi_action(predictions, vectorized=True, mask=masks)
                else:
                    actions = prediction_best(predictions, vectorized=True, mask=masks)
        actions_iter = zip(*actions) if self.is_multi_action else iter(actions)
        for i, (sim, action) in enumerate(zip(self.sims, actions_iter)):
            if sim.is_terminal():
                continue
            if self.is_solo_env() or self._is_training_agent_turn(sim, i) == training_agent:
                sim._step(action)
        del predictions, actions_iter
        return X, actions
    def is_solo_env(self) -> bool:
        return self.fixed_model is None
    def _play_until_turn_over(self, training_agent: bool):
        if not training_agent and self.is_solo_env(): return
        while any(self._is_training_agent_turn(sim, i) == training_agent for i, sim in enumerate(self.sims) if not sim.is_terminal()):
            self._play(training_agent)
        if self.step_delay_millis > 0:
            sleep(self.step_delay_millis * 0.001)
    def _is_training_agent_turn(self, sim: DuelGame, sim_index: int) -> bool:
        return sim.get_turn() == sim_index % 2
    def _iteration_step(self):

        # 1. Initialize transition info.
        rewards = np.empty(self.num_sims)
        terminals = np.zeros(self.num_sims, dtype=bool)
        for i, sim in enumerate(self.sims):
            rewards[i] = -sim.get_static_reward()

        # 2. The training agent plays a single move.
        old_states, actions = self._play(True)

        # 3. The fixed agent plays enough moves to come back to the training agent's turn (or end the game).
        self._play_until_turn_over(False)

        # 4. Record transition info.
        new_states = self._get_model_inputs()
        for i, sim in enumerate(self.sims):
            rewards[i] += sim.get_static_reward()
            if i % 2 == 1:
                rewards[i] *= -1
            sim.accumulated_reward += rewards[i]
            if sim.is_terminal():
                terminals[i] = True
                self.statistics.record('train_reward_per_episode', sim.accumulated_reward, max_length=self.stats_max_length)
        if not self.is_multi_action:
            masks = self._get_action_masks()
        self.transition_memory.record_transitions(old_states, new_states, rewards, actions, terminals, masks)
        self.statistics.record('train_reward_per_step', np.mean(rewards), max_length=self.stats_max_length)

        # 5. Reset finished environments.
        for i, sim in enumerate(self.sims):
            if terminals[i]:
                sim._reset()
        self._play_until_turn_over(False)
        del old_states, new_states, rewards, actions, terminals, masks

        # 6. Perform experience replay.
        do_save_model = self.iteration_steps % self.model_save_interval == 0
        if self.iteration_steps % self.experience_replay_interval == 0:
            for _ in range(self.experience_replays_per_interval):
                experience_replay(self.training_model, int(self.experience_replay_sample_rate * self.num_sims + .5), self.transition_memory,
                    fit_epochs=self.fit_epochs,
                    predict_batch_size=self.predict_batch_size,
                    fit_batch_size=self.fit_batch_size,
                    gamma=self.gamma,
                    tau=self.tau,
                    verbose=False,
                    save_target_model=do_save_model,
                )
        if do_save_model:
            self.save_training_model()
            self.save_training_history_data()

        self.iteration_steps += 1
        self.total_iteration_steps += 1
    
    def set_step_delay(self, delay_millis: float):
        self.step_delay_millis = delay_millis

    def run_iteration(self, steps: int):
        '''
        Perform one iteration of training.  The model plays against the previous version of itself for a fixed number of total steps.  If this is the first iteration, then the opposing agent plays random moves.

        The total number of transitions generated for training after running the training iteration is equal to

        ```
        total_transitions = steps * num_sims
        ```

        where `num_sims` is defined in the constructor of `DuelEnv`.
        '''

        if self.training_model is None:
            raise Exception(f'Please set models with DuelEnv.set_models().')

        self.iteration_steps = 0
        if not self.is_solo_env():
            self.fixed_model.save(f'{self.model_save_dir}/opponent_iter{self.total_iterations}.h5')
        for sim in self.sims:
            sim._reset()
        self._play_until_turn_over(False)
        print('Sample of model prediction on 8 inputs taken from a normal distribution.  If the predictions are equivalent, then ReLU needs to be replaced with LeakyReLU.')
        print(self.training_model(np.random.normal(0, 1, (8, 14, 7))).numpy())
        collected = gc.collect()
        print(f'[Garbage Collector] Collected {collected} objects.')
        with StylizedProgressBar(f'Iteration #{self.total_iterations + 1}', max=steps) as bar:
            for _ in range(steps):
                self._iteration_step()
                bar.next()
        if not self.is_solo_env():
            self.fixed_model.set_weights(self.training_model.get_weights())
        self.transition_memory.clear()
        self.total_iterations += 1

    def run(self, steps_per_iteration: int = 2048):
        '''
        Repeatedly call `DuelEnv.run_iteration(steps_per_iteration)`, and track which model is the best from each iteration, and then use that best model as the fixed model.  Additionally, print useful information about the performance of each model between each iteration, obtained by simulating tournaments between all models.

        This function saves the best-performing model in `DuelEnv.model_save_dir` as `best.h5`.

        It is best to have a large `steps_per_iteration`, but too large will result in overfitting toward the opponent, resulting in less generalized behavior and strategy.
        '''
        while True:
            self.run_iteration(steps_per_iteration)
            self.print_statistics()

            self.save_training_model()
            
            _, _, fnames = next(os.walk(self.model_save_dir))
            fnames = list(filter(lambda s: s.endswith('.h5') and not s.endswith('best.h5'), fnames))
            if len(fnames) > 16:
                fnames = fnames[-16:]
            if not any('current' in fname for fname in fnames):
                fnames.append('current.h5')
            fpaths = [
                f'{self.model_save_dir}/{fname}'
                for fname in fnames
            ]
            model_names = fpaths
            num_pairings = number_of_pairings(len(fpaths))
            model_ratings = load_model_ratings(self.model_save_dir)
            results, stats = tournament(self.game_cls,
                model_fpaths=fpaths,
                model_names=model_names,
                model_ratings=model_ratings,
                games_per_pairing=(4 * self.predict_batch_size // num_pairings) // 2 * 2,
                predict_batch_size=self.predict_batch_size,
                workers=4,
                verbose=True,
                record_statistics=True,
            )
            save_model_ratings(self.model_save_dir, model_ratings)
            print('Round-robin results:')
            print(tournament_results_str(results))
            score_per_model = get_score_per_model(results)
            print('Model scores:')
            print(model_scores_str(score_per_model))
            print('Model ratings:')
            print(model_ratings)
            best_model_name = max(model_ratings.items(), key=lambda t: t[1].mu-t[1].sigma)[0]
            print('Best model:', best_model_name)
            print('Using best model as fixed model in training.')
            best_model = load_model(best_model_name)
            best_model.save(f'{self.model_save_dir}/best.h5')
            self.set_models(self.training_model, best_model)

            # update metrics
            self.statistics.extend(stats[f'{self.model_save_dir}/current.h5'])
            self.save_training_history_data()



