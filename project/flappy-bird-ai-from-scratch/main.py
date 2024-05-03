import pygame
import config
import componenets
import population
from sys import exit

pygame.init()
clock = pygame.time.Clock()
population = population.Population(n=10)

def generate_pipes():
    config.pipes.append(componenets.Pipes(config.WIDTH))

def quit_game():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

def main():
    pipes_spawn_time = 10
    while True:
        quit_game()
        config.window.fill((0, 0, 0))

        # Spawn ground
        config.ground.draw(config.window)

        # Spawn pipes
        if pipes_spawn_time <= 0:
            generate_pipes()
            pipes_spawn_time = 200
        pipes_spawn_time -=1 

        for pipe in config.pipes:
            pipe.draw(config.window)
            pipe.update()
            if pipe.off_screen:
                config.pipes.remove(pipe)
                
        # Spawn players
        if not population.extinct():
            population.update_live_players() 
        else: pass
        
        clock.tick(60)
        pygame.display.flip()


if __name__ == "__main__":
    main()
