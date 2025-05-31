"""
Reasoning and Decision Making under Uncertainty

Portfolio 2 - Particle Filter

Peter Preinesberger
Illia Rohalskyi
"""
import random as rd
import pygame
import numpy as np
from World.WorldInformation import BallWorldInformation
from World.Process import BallArenaProcess, StochasticBallArenaProcess
from World.Initializer import RandomBallInitializer, UniformPositionNormalVelocityInitializer
from Filter.Observation import MultiBallObservationModel
from Filter import ParticleSet
from Sensor import MultiBallSensor

import line_profiler

def main():
    N_BALLS = 5
    MPS = 60
    
    colors = [[rd.randint(0,255),rd.randint(0,255),rd.randint(0,255)] for _ in range(N_BALLS)]

    world = BallWorldInformation(
        width = 50,
        height = 50,
        gravity = 9.8,
        ball_radius = 1,
        bounce_discount = 0.8,
        air_discount = 0.9,
        ground_discount = 0.5
    )

    I = UniformPositionNormalVelocityInitializer(
        np.diag(np.array([100,100]).astype(float)),
        world
    )
    states: list[np.ndarray] = [I.generate(n) for n in range(N_BALLS)]
    
    
    sensor_variance = np.array([0.1,0.1]).astype(float)
    sensor = MultiBallSensor(np.diag(sensor_variance), seed = 0)
    
    P: BallArenaProcess = BallArenaProcess(world)
    
    # particle filter initialisation
    assumed_world = world
    assumed_transition_process = StochasticBallArenaProcess(
        BallArenaProcess(assumed_world),
        np.array([10,10]).astype(float) # non-determinism in transition model
    )
    assumed_observation_variance = np.array([100,100]).astype(float)
    observation_model = MultiBallObservationModel(assumed_observation_variance)
    n_particles = 10000
    seed = 1337
    assumed_initialization = UniformPositionNormalVelocityInitializer(
        np.diag(np.array([1,1]).astype(float)),
        assumed_world
    )
    
    
    particle_set: ParticleSet = ParticleSet(
        n_particles,
        assumed_initialization,
        assumed_transition_process,
        observation_model,
        seed
    )    

    DIM = 1000
    MARGIN = 0.1
    screen = pygame.display.set_mode((DIM, DIM))
    clock = pygame.time.Clock()
    running = True
    
    INNER = DIM - 2 * DIM * MARGIN
    BORDER = MARGIN * DIM
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
                screen.fill("black")
                
        observations = sensor.sense(states)    
                
        ###
                
        # draw arena boundary
        pygame.draw.rect(screen, "grey", [BORDER, BORDER, INNER, INNER])
                
        for (ball_num, ball) in enumerate(states):
            pos = ball[:2]
            pos_x = (ball[0] / world.width) * INNER + BORDER
            pos_y = INNER - (ball[1] / world.height) * INNER + BORDER
            rad = (world.ball_radius / world.width) * INNER
            pygame.draw.circle(screen, colors[ball_num], [pos_x, pos_y], rad)
                    
        for (ball_num, ball) in enumerate(observations):
            pos_x = (ball[0] / world.width) * INNER + BORDER
            pos_y = INNER - (ball[1] / world.height) * INNER + BORDER
            rad = 5
            pygame.draw.circle(screen, "black", [pos_x, pos_y], rad)

        ###
        ma = (max(particle_set.weights))
        mi = (min(particle_set.weights))
        for (p,w) in zip(particle_set.particles, particle_set.weights):
            pos = p[:2]
            pos_x = (pos[0] / world.width) * INNER + BORDER
            pos_y = INNER - (pos[1] / world.height) * INNER + BORDER
            rad = 3
            coeff = (w - mi) / max((ma - mi),0.0001)
            pygame.draw.circle(screen, [coeff*255,  coeff*255, 0], [pos_x, pos_y], rad)

        ### do the particle filter
        
        particle_set.resample()
        particle_set.transition(delta = 1/MPS)
        

        particle_set.observe(observations)

        ###
        pygame.display.flip()

        states = P.transition(states, 1/MPS)
        clock.tick(60)


        
pygame.init()
main()
pygame.quit()
