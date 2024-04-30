import pygame
import sys
import time

# Constants
WIDTH = 1000
HEIGHT = 600
PLAYER_WIDTH = 140
PLAYER_HEIGHT = 100
OBSTACLE_WIDTH = 60
OBSTACLE_HEIGHT = 80
BACKGROUND_COLOR = (255, 255, 255)
PLAYER_SPEED = 20
GRAVITY = 0.6
JUMP_HEIGHT = 15
MAX_JUMP_COUNT = 5

# Background music
pygame.mixer.init()
pygame.mixer.music.load('./audio/background-music.wav')
pygame.mixer.music.play(-1)  # -1 for looping

# Jumping sound effect
jump_sound = pygame.mixer.Sound('./audio/jump-sound.wav')

# Initialize additional variables
cactus_spawn_rate = 1  # Initial cactus spawn rate
time_elapsed = 0  # Variable for time elapsed
successful_avoidance = 0  # Variable for successful cactus avoidance

# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Dino Game")

# Load character images
player_running_image = pygame.image.load('./images/cat-running-sprite.png').convert_alpha()
player_jump_image = pygame.image.load('./images/cat-jumping-sprite.png').convert_alpha()
obstacle_image = pygame.image.load('./images/cactus-sprite.png').convert_alpha()

# Initialize time and score variables
start_time = time.time()
score = 0

clock = pygame.time.Clock()

class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image_running = pygame.transform.scale(player_running_image, (PLAYER_WIDTH, PLAYER_HEIGHT))
        self.image_jump = pygame.transform.scale(player_jump_image, (PLAYER_WIDTH, PLAYER_HEIGHT))
        self.image = self.image_running
        self.rect = self.image.get_rect()
        self.rect.x = 50
        self.rect.y = HEIGHT - PLAYER_HEIGHT
        self.velocity = 0
        self.jumping = False
        self.jump_count = 0  # Number of consecutive jumps made

    def update(self):
        self.velocity += GRAVITY
        self.rect.y += self.velocity

        if self.rect.y >= HEIGHT - PLAYER_HEIGHT:
            self.rect.y = HEIGHT - PLAYER_HEIGHT
            self.velocity = 0
            self.jumping = False
            self.jump_count = 0  # Reset jump count when landing

        if self.jumping:
            self.image = self.image_jump
        else:
            self.image = self.image_running

    def jump(self):
        if self.rect.y >= PLAYER_HEIGHT + 100: # Check if max jump count is not reached
            self.velocity = -JUMP_HEIGHT
            self.jumping = True


class Obstacle(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.transform.scale(obstacle_image, (OBSTACLE_WIDTH, OBSTACLE_HEIGHT))
        self.rect = self.image.get_rect()
        self.rect.x = WIDTH
        self.rect.y = HEIGHT - OBSTACLE_HEIGHT

    def update(self):
        self.rect.x -= PLAYER_SPEED

# Create sprites groups
all_sprites = pygame.sprite.Group()
obstacles = pygame.sprite.Group()

# Create player
player = Player()
all_sprites.add(player)

# Main game loop
running = True
game_over = False
while running:
    current_time = time.time() - start_time  # Calculate current time
    minutes = int(current_time // 60)
    seconds = int(current_time % 60)
    score_increment = 3 * (minutes + 1)  # Calculate score increment based on time

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if not game_over:
                if event.key == pygame.K_SPACE:
                    player.jump()
                    jump_sound.play()  # Play jump sound when the player jumps
            else:  # If game is over
                if event.key == pygame.K_y:
                    # Reset game state to play again
                    start_time = time.time()
                    score = 0
                    time_elapsed = 0  # Reset time elapsed
                    successful_avoidance = 0  # Reset successful avoidance
                    game_over = False
                    player.rect.y = HEIGHT - PLAYER_HEIGHT  # Reset player position
                    obstacles.empty()  # Remove all obstacles
                    # Add any other initialization/reset logic here
                elif event.key == pygame.K_q:
                    running = False  # Quit the game

    if not game_over:
        # Generate obstacles
        if pygame.time.get_ticks() % (300 - (20 * minutes)) == 0:  # Adjust obstacle spawning speed based on time
            obstacle = Obstacle()
            all_sprites.add(obstacle)
            obstacles.add(obstacle)

        # Update sprites
        all_sprites.update()

        # Check for collisions
        hits = pygame.sprite.spritecollide(player, obstacles, False)
        if hits:
            game_over = True

        # Update time elapsed
        time_elapsed = current_time

        # Update successful cactus avoidance
        if hits:
            successful_avoidance += 1

        # Adjust cactus spawn rate based on time
        if int(current_time) % 10 == 0:  # Increase cactus spawn rate every 10 seconds
            cactus_spawn_rate += 1

        # Generate obstacles based on cactus spawn rate
        if pygame.time.get_ticks() % (300 - (20 * cactus_spawn_rate)) == 0:
            obstacle = Obstacle()
            all_sprites.add(obstacle)
            obstacles.add(obstacle)

        # Update scoring system to incorporate time elapsed and successful avoidance
        score = 3 * time_elapsed + 10 * successful_avoidance

        # Draw everything
        screen.fill(BACKGROUND_COLOR)
        all_sprites.draw(screen)

        # Render text for time and score
        pygame.font.init()
        font_path = "./nasalization-rg.otf"
        font = pygame.font.Font(font_path, 36)
        time_text = font.render(f"Time: {minutes:02}m {seconds:02}s", True, (0, 0, 0))
        score_text = font.render(f"Score: {int(score)}", True, (0, 0, 0))

        # Display time and score
        screen.blit(score_text, (WIDTH - 310, 60))
        screen.blit(time_text, (WIDTH - 310, 20))

        # Update display
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(60)
    else:
        # Game over screen
        game_over_text = font.render("OOPS! YOU ARE DEAD ~", True, (255, 0, 0))
        restart_text = font.render("Press 'Y' to play again", True, (0, 0, 255))
        quit_text = font.render("Press 'Q' to quit the game", True, (0, 0, 255))
        screen.blit(game_over_text, (WIDTH // 2 - 200, HEIGHT // 2 - 50))
        screen.blit(restart_text, (WIDTH // 2 - 200, HEIGHT // 2 + 20))
        screen.blit(quit_text, (WIDTH // 2 - 200, HEIGHT // 2 + 60))
        pygame.display.flip()

pygame.quit()
sys.exit()
