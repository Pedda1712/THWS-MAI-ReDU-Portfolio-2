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
from World.Process import BallArenaProcess
from World.Initializer import RandomBallInitializer

I = RandomBallInitializer(np.array([25, 25, 0, 10]), np.diag([3, 3, 100, 100]))

N_BALLS = 10
MPS = 60

states: list[np.ndarray] = [I.generate(n) for n in range(N_BALLS)]
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

P: BallArenaProcess = BallArenaProcess(world)

pygame.init()
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
    
    ###

    # draw arena boundary
    pygame.draw.rect(screen, "grey", [BORDER, BORDER, INNER, INNER])
    
    for (ball_num, ball) in enumerate(states):
        # transform ball coordinates to image space and draw
        pos = ball[:2]
        pos_x = (ball[0] / world.width) * INNER + BORDER
        pos_y = INNER - (ball[1] / world.height) * INNER + BORDER
        rad = (world.ball_radius / world.width) * INNER
        pygame.draw.circle(screen, colors[ball_num], [pos_x, pos_y], rad)

    ###
    pygame.display.flip()

    states = P.transition(states, 1/MPS)
    clock.tick(60)

pygame.quit()


