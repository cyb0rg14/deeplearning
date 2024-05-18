import sys
import pygame
import numpy as np
from constants import *

# intialize pygame
pygame.init()

# create screen 
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("TIC TAC TOE AI")
screen.fill(BG_COLOR)

# create board class
class Board:
    def __init__(self) -> None:
        self.squares = np.zeros((ROWS, COLS))

    def mark_sqr(self, row: int, col: int, player: int) -> None:
        self.squares[row][col] = player

    def check_empty_sqr(self, row: int, col: int) -> bool:
        return self.squares[row][col]  == 0
    

# create game class
class Game:
    def __init__(self) -> None:
        self.board = Board()
        self.player = 1
        self.draw_lines()
        
    def draw_lines(self):
        # vertical lines  
        pygame.draw.lines(screen, LINE_COLOR, points=[(SQSIZE, 0), (SQSIZE, HEIGHT)], width=LINE_WIDTH, closed=True)
        pygame.draw.lines(screen, LINE_COLOR, points=[(WIDTH - SQSIZE, 0), (WIDTH - SQSIZE, HEIGHT)], width=LINE_WIDTH, closed=True)

        # horizontal lines
        pygame.draw.lines(screen, LINE_COLOR, points=[(0, SQSIZE), (WIDTH, SQSIZE)], width=LINE_WIDTH, closed=True)
        pygame.draw.lines(screen, LINE_COLOR, points=[(0, HEIGHT - SQSIZE), (WIDTH, HEIGHT - SQSIZE)], width=LINE_WIDTH, closed=True)
        
    def next_turn(self):
        self.player = self.player % 2 + 1

    def draw_figure(self, row: int, col: int):
        if self.player == 1:
            pass
        elif self.player == 2:
            center = (col * SQSIZE + SQSIZE // 2, row * SQSIZE + SQSIZE // 2)
            pygame.draw.circle(
                screen, CIRC_COLOR, center, RADIUS, CIRC_WIDTH
            )

# main function
def game():
    gameInstance = Game()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit() 
                
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = event.pos
                row = pos[0] // SQSIZE
                col = pos[1] // SQSIZE

                if gameInstance.board.check_empty_sqr(row, col):
                    gameInstance.board.mark_sqr(row, col, gameInstance.player)
                    gameInstance.draw_figure(row, col)
                    gameInstance.next_turn()

                
        pygame.display.update()
                
# entry point
if __name__ == "__main__":
    game()
