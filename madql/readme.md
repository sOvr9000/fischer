<h2>Instructions</h2>

This is a generalized deep Q learning interface for multi-agent environments. Create an environment, add agents via <code>add_agent()</code>, and then repeatedly call <code>step()</code> to simulate the environment.  Every other part of the algorithm is handled automatically.

Implement custom dynamics of the environment by overwriting <code>on_step()</code>, <code>do_actions()</code>, <code>get_observation()</code>, <code>get_reward()</code>, and <code>is_state_terminal()</code> in <code>RLEnvironment</code>.

Configure a unique Keras model for each agent, or let some or all agents share a Keras model. An agent can be set to take random actions with uniform distribution from the action space.

Specify <code>state_shape</code> and <code>action_shape</code> in the constructor of <code>RLEnvironment</code>. The input and output shapes of an agent's Keras model should coincide with these parameters.

The rest of the training process is set up automatically. Be sure to tune some of the hyperparameters for training with <code>set_parameters()</code>.


