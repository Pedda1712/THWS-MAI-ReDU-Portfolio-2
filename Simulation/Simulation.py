"""
The experiment "engine".
"""
import random as rd
import pygame
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional

from World.WorldInformation import BallWorldInformation
from World.Process import BallArenaProcess, StochasticBallArenaProcess
from World.Initializer import RandomBallInitializer, UniformPositionNormalVelocityInitializer
from Filter.Observation import MultiBallObservationModel
from Filter import ParticleSet, BallEstimator
from Sensor import MultiBallSensor
from .SimulationParameters import SimulationParameters

class Simulation:
    p: SimulationParameters

    def __init__(self, p: SimulationParameters):
        self.p = p

    def run(self):
        # the actual world
        world = BallWorldInformation(
            width = self.p.width,
            height = self.p.height,
            gravity = self.p.gravity,
            ball_radius = self.p.ball_radius,
            bounce_discount = self.p.bounce_discount,
            air_discount = self.p.air_discount,
            ground_discount = self.p.ground_discount
        )

        initializer = UniformPositionNormalVelocityInitializer(
            np.diag(self.p.initial_velocity_variance).astype(float),
            world
        )

        states: list[np.ndarray] = [initializer.generate(n) for n in range(self.p.number_of_balls)]

        sensor = MultiBallSensor(
            np.diag(self.p.sensor_variance).astype(float),
            seed = self.p.seed
        )

        process: BallArenaProcess = BallArenaProcess(world)

        # our assumptions about the world
        assumed_world = BallWorldInformation(
            width = self.p.assumed_width,
            height = self.p.assumed_height,
            gravity = self.p.assumed_gravity,
            ball_radius = self.p.assumed_ball_radius,
            bounce_discount = self.p.assumed_bounce_discount,
            air_discount = self.p.assumed_air_discount,
            ground_discount = self.p.assumed_ground_discount
        )

        assumed_deterministic_process = BallArenaProcess(assumed_world)
        
        assumed_transition_process = StochasticBallArenaProcess(
            assumed_deterministic_process,
            np.array(self.p.transition_velocity_variance).astype(float)
        )

        observation_model = MultiBallObservationModel(
            np.array(self.p.assumed_sensor_variance).astype(float)
        )

        assumed_initialization = UniformPositionNormalVelocityInitializer(
            np.diag(self.p.assumed_initial_velocity_variance).astype(float),
            assumed_world
        )

        particle_set: ParticleSet = ParticleSet(
            self.p.number_of_particles,
            assumed_initialization,
            assumed_transition_process,
            assumed_deterministic_process,
            observation_model,
            self.p.seed
        )

        est = BallEstimator()

        # pygame window parameters
        DIM = 1000
        MARGIN = 0.1
        INNER = DIM - 2 * DIM * MARGIN
        BORDER = MARGIN * DIM

        running = True
        observation_missing = False
        
        screen: list[pygame.Surface] = []
        clock = []
        my_font = []
        
        if self.p.live_show:
            pygame.init()
            pygame.font.init() 
            my_font = [pygame.font.SysFont('Comic Sans MS', 30)]
            screen = [pygame.display.set_mode((DIM,DIM))]
            clock = [pygame.time.Clock()]

        # previously seen states
        states_backlog = []
        est_states_backlog = []

        states_history = []
        estimated_states_history = []

        steps = 0
        while running:
            # sense current state
            observations = sensor.sense(states)
            
            if not observation_missing:
                estimated_states = est.estimate(
                    self.p.assumed_number_of_balls,
                    particle_set
                )
            else:
                # if the observation is missing, just propagate old estimates
                estimated_states = assumed_deterministic_process.transition(estimated_states, 1 / self.p.measurements_per_second)
            
            if not observation_missing:
                # Condensation Algorithm
                particle_set.resample()
                particle_set.transition(1 / self.p.measurements_per_second, deterministic = observation_missing)
                particle_set.observe(observations)
            else:
                # propagate the particles deterministically in case of missing observation
                particle_set.transition(1 / self.p.measurements_per_second, deterministic = observation_missing)
                
            states_history.append(states)
            estimated_states_history.append(estimated_states)
            states_backlog.append(states)
            est_states_backlog.append(estimated_states)
            while len(est_states_backlog) > self.p.visualize_tail_length:
                est_states_backlog.pop(0)
                states_backlog.pop(0)

            # actual state update
            states = process.transition(states, 1 / self.p.measurements_per_second)
            
            if self.p.live_show:
                # Everything in here is only drawing code
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_d:
                            observation_missing = True
                    elif event.type == pygame.KEYUP:
                        if event.key == pygame.K_d:
                            observation_missing = False                    
                screen[0].fill("black")
                text_surface = my_font[0].render("press 'd' to make observations cut out", False, (255, 0, 0))
                screen[0].blit(text_surface, (10,10))
                pygame.draw.rect(screen[0], "grey", [BORDER, BORDER, INNER, INNER])
                if self.p.show_actual_positions:
                    for (i, _states) in enumerate(states_backlog):
                        for (ball_num, ball) in enumerate(_states):
                            pos_x = (ball[0] / world.width) * INNER + BORDER
                            pos_y = INNER - (ball[1] / world.height) * INNER + BORDER
                            rad = (world.ball_radius / world.width) * INNER
                            pygame.draw.circle(screen[0], (0,0,int(255 * i/self.p.visualize_tail_length)), [pos_x, pos_y], rad)

                for (i, _states) in enumerate(est_states_backlog):
                    for (ball_num, ball) in enumerate(_states):
                        pos_x = (ball[0] / world.width) * INNER + BORDER
                        pos_y = INNER - (ball[1] / world.height) * INNER + BORDER
                        rad = (assumed_world.ball_radius / world.width) * INNER
                        pygame.draw.circle(screen[0], (0,int(255 * i/self.p.visualize_tail_length),0), [pos_x, pos_y], rad)

                if self.p.show_observations:
                    for (ball_num, ball) in enumerate(observations):
                        pos_x = (ball[0] / world.width) * INNER + BORDER
                        pos_y = INNER - (ball[1] / world.height) * INNER + BORDER
                        rad = 5
                        pygame.draw.circle(screen[0], "red", [pos_x, pos_y], rad)

                if self.p.show_particles:
                    ma = (max(particle_set.weights))
                    mi = (min(particle_set.weights))
                    for (p,w) in zip(particle_set.particles, particle_set.weights):
                        pos = p[:2]
                        pos_x = (pos[0] / world.width) * INNER + BORDER
                        pos_y = INNER - (pos[1] / world.height) * INNER + BORDER
                        rad = 3
                        coeff = (w - mi) / max((ma - mi),0.0001)
                        pygame.draw.circle(screen[0], (int(coeff*255),  int(coeff*255), 0), [pos_x, pos_y], rad)

                pygame.display.flip()
                clock[0].tick(60)

            steps += 1
            if steps > self.p.max_steps:
                running = False

        if self.p.show_summary_plots:
            # only drawing code in here
            a_states_history = np.array(states_history)
            a_estimated_states_history = np.array(estimated_states_history)
            labels = ["x position over time", "y position over time", "x velocity over time", "y velocity over time"]
            axlabels = ["x","y","vx","vy"]

            fig, axs = plt.subplots(2, 2)
            fig.suptitle("actual (blue) vs estimated (green) parameters")

            for (dim,ax) in zip(range(a_states_history.shape[2]), axs.flat):
                ax.set_title(labels[dim])
                ax.set_xlabel("Time Step")
                ax.set_ylabel(axlabels[dim])
                for nball in range(a_states_history.shape[1]):
                    ameas = a_states_history[:,nball,dim]
                    ax.plot(ameas, "bo", markersize=2)
                for eball in range(a_estimated_states_history.shape[1]):
                    bmeas = a_estimated_states_history[:,eball,dim]
                    ax.plot(bmeas, "go", markersize=2)
            
            plt.show()
                
        if self.p.live_show:
            pygame.quit()
