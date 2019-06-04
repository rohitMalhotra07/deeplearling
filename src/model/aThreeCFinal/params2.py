class Params():
    def __init__(self,params):
        self.lr = 0.0001
        self.gamma = params.gamma
        self.tau = 1.
        self.seed = 1
        self.num_processes = params.num_agents_a3c
        self.num_steps = 20
        #self.max_episode_length = 10000 Using no maximum episode length