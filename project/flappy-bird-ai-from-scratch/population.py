import config
import player

class Population:
    def __init__(self, n):
        self.players = []
        self.n = n
        for _ in range(self.n):
            self.players.append(player.Player())

    def update_live_players(self):
        for player in self.players:
            if player.alive:
                player.think()
                player.draw(config.window)
                player.update(config.ground)
        
    def extinct(self):
        extinct = True
        for player in self.players:
            if player.alive:
                extinct = False
        return extinct
