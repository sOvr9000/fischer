
import os
from keras.models import Model
from fischer.dql.duelenv import DuelEnv
from fischer.dql import Stats, TransitionMemory
from fischer.dt import dt
from .sologame import SoloGame



__all__ = ['SoloEnv']

class SoloEnv(DuelEnv):
    def __init__(self,
        game_cls: type[SoloGame],
        num_sims: int,
        memory_capacity: int,
        
        fit_epochs: int = 1,
        predict_batch_size: int = 128,
        fit_batch_size: int = 32,
        gamma: float = 0.98,
        tau: float = 0.01,
        exploration_interval: int = 256, # number of steps per exploration-exploitation cycle (beginning of iteration is exploration, gradually approach exploitation, but loop back to exploration cyclically over the duration of the iteration); in general, this should be greater than or equal to the number of steps per episode
        heuristic_interval: int = 384, # like exploration_interval, but this is for when the agent selects an action decided by DuelGame.get_heuristic_action(), which is meant to be a best-guess; heuristic selection overrides random selection

        experience_replay_sample_rate: float = 0.75,
        experience_replay_interval: int = 16,

        model_save_interval: int = 32,
        model_save_dir: str = None,
    ):
        # super().__init__(
        #     game_cls=game_cls,
        #     num_sims=num_sims,
        #     memory_capacity=memory_capacity,
        #     fit_epochs=fit_epochs,
        #     predict_batch_size=predict_batch_size,
        #     fit_batch_size=fit_batch_size,
        #     gamma=gamma,
        #     tau=tau,
        #     experience_replay_sample_rate=experience_replay_sample_rate,
        #     experience_replay_interval=experience_replay_interval,
        #     model_save_interval=model_save_interval,
        #     model_save_dir=model_save_dir,
        # )

        assert issubclass(game_cls, SoloGame)
        assert isinstance(num_sims, int) and num_sims > 0

        self.sims: list[SoloGame] = [
            game_cls()
            for _ in range(num_sims)
        ]
        for i in range(num_sims):
            self.sims[i].sim_index = i
        self.num_sims = num_sims
        self.statistics = Stats()
        self.memory_capacity = memory_capacity
        self.total_iterations = 0
        self.total_iteration_steps = 0
        self.step_delay_millis = 0

        self.fit_epochs = fit_epochs
        self.predict_batch_size = predict_batch_size
        self.fit_batch_size = fit_batch_size
        self.gamma = gamma
        self.tau = tau
        self.exploration_interval = exploration_interval
        self.heuristic_interval = heuristic_interval
        self.step_delay_millis = 0

        self.experience_replay_sample_rate = experience_replay_sample_rate
        self.experience_replay_interval = experience_replay_interval

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
    def set_models(self, training_model: Model):
        if hasattr(self, 'transition_memory'):
            del self.transition_memory
        if isinstance(training_model.input_shape, list):
            self.transition_memory = TransitionMemory(capacity=self.memory_capacity, state_shape=[s[1:] for s in training_model.input_shape], action_shape=training_model.output_shape[1:])
        else:
            self.transition_memory = TransitionMemory(capacity=self.memory_capacity, state_shape=training_model.input_shape[1:], action_shape=training_model.output_shape[1:])
        self.training_model = training_model
        self.fixed_model = None
        self.is_multi_action = isinstance(self.training_model.output_shape, list)
    def set_model(self, training_model: Model):
        self.set_models(training_model)
    def run_iteration(self, steps: int):
        '''
        Perform one iteration of training.  The model plays for a fixed number of total steps.

        The total number of transitions generated for training after running the training iteration is equal to

        ```
        total_transitions = steps * num_sims
        ```

        where `num_sims` is defined in the constructor of `SoloEnv`.
        '''
        if self.training_model is None:
            raise Exception(f'Please set a model with SoloEnv.set_model().')
        super().run_iteration(steps)
