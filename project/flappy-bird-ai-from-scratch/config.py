import pygame
import componenets

WIDTH = 550
HEIGHT = 720
window = pygame.display.set_mode((WIDTH, HEIGHT))

ground = componenets.Ground(WIDTH)
# pipes = componenets.Pipes(WIDTH)
pipes = []
