import pygame
import random

class Ground:
    ground_level = 500
    def __init__(self, win_width):
        self.x, self.y = 0, Ground.ground_level
        self.rect = pygame.Rect(self.x, self.y, win_width, 5)

    def draw(self, window):
        pygame.draw.rect(window, (255, 255, 255), self.rect)

class Pipes:
    width = 15
    opening = 100
    def __init__(self, win_width):
        self.x = win_width
        self.bottom_height = random.randint(10, 300)
        self.top_height = Ground.ground_level - self.bottom_height - self.opening
        self.bottom_rect, self.top_rect = pygame.Rect(0, 0, 0, 0), pygame.Rect(0, 0, 0, 0)
        self.passed = False
        self.off_screen = False

    def draw(self, window):
        self.bottom_rect = pygame.Rect(self.x, Ground.ground_level - self.bottom_height, self.width, self.bottom_height)
        pygame.draw.rect(window, (255, 255, 255), self.bottom_rect)
        self.top_rect = pygame.Rect(self.x, 0, self.width, self.top_height) 
        pygame.draw.rect(window, (255, 255, 255), self.top_rect)

    def update(self):
        self.x -= 1
        if self.x + Pipes.width <= 50:
            self.passed = True
        if self.x <= -self.width:
            self.off_screen = False
