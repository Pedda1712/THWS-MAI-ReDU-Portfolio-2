# Reasoning and Decision Making under Uncertainty : Portfolio 2
Authors: *Peter Preinesberger* and *Illia Rohalskyi*.

The Task:
Realize an implementation of the Particle Filter in a programming language of your choice for a
simulation of the ball-throwing example from the lecture slides. The task of your Particle Filter is
to estimate the positions and velocity vectors of n ≥ 1 balls ying simultaneously only from the
observed erroneous positions over time.

## Thoughts on State and Observation Model
The state in our case contains the position and velocity of one ball. The division into multiple balls is handled in the evaluation step.

The evaluation step enforces a constraint, where for N observations, each observation is allowed to 'give out' 1/N weight. An observation
will distribute its weight to the particles that have it as its nearest observation according to normal distributions centered at the particle positions. In this way, the resampling step will (in expectation) generate an equal amount of particles for each observation.

*Note*: If this partitioning step is skipped, and we simply sum the values of the normal distributions at the observations for each
particle, we will wind up with unequal particle counts for each ball. In our observations, this situation quickly degenerates to
having all particles allocated for one ball. This makes sense from the theoretical perspective: Only summing the normal distributions
is equivalent to the assumption of having only *one* actual ball and N observations of it: Over time, the particle cloud will move to the
most likely position instead of dividing itself amongst the N most likely positions.

## Code Architecture
Here, we will briefly review the purpose of the individual code modules and sketch their composition.

### World
This module contains initialization logic and state transition functions. 

In particular, the **Initializer** subpackage contains the ball initialization strategies.

The **Process** subpackage contains state transition functions. These state transitions will be used to simulate the actual physical world as well as serve as the state transition model in the particle filter (potentially different parameters in each case).

There is no actual World 'object': We only provide the facilities (initialization & transition) to create one here. 

### Sensor
The sensor takes in an actual state, adds some parametrized noise onto the ball position and returns the noisy positions. 

### Filter
The **Observation** subpackage implements the evaluation step of the condensation algorithm as described above.

The ```ParticleSet``` class is the actual Particle Filter implementation. Because we divided our World into initialization and transition classes, the particle filter can use the same code for the transition as the world. Note that we only use the code: The ParticleSet contains a transition object that captures what we *assume* about the environment (can differ from the transition used in the actual world). In particular, the ParticleSet will use a ```StochasticBallArenaProcess```, that adds noise onto the velocity before transition to enable hypothesis exploration.

The ```BallEstimator``` will estimate N ball positions and velocities from a set of particles by utilizing KMeans clustering on the particle positions from the particle set.

### Simulation
The ```Simulation``` class orchestrates the entire process: It will initialize true ball positions and transition them each step with a ```BallArenaProcess``` instance. It will generate observations from the true states by using ```MultiBallSensor```, and run the four steps of the ```ParticleSet ```. The ```ParticleSet``` uses a ```StochasticBallArenaProcess``` with the assumed world parameters and parametrizable non-determinism. Finally, ```BallEstimator``` is used to fetch ball positions and velocities from the particle filter at each step.

Each step is visualized using PyGame, and summary plots are generated at the end of the experiment.

## Running
Required packages: ```numpy matplotlib pygame```.

Run the ```__gui__.py``` script to be able to tune parameters and visualize the results.
