from snake.game import Game, GameConf, GameMode

# RL algorithm
greedy = "GreedySolver"
hamilton = "HamiltonSolver"

# game mode
normal = GameMode.NORMAL

conf = GameConf() # initialize game conf which contains most of the game's logic and functionality
conf.solver_name = hamilton # what solver to use
conf.mode = normal   # game mode
print("Solver: %s " % (conf.solver_name))
print("Mode: %s" %(conf.mode))
Game(conf).run()
