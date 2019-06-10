import torch
import os
import numpy as np
from logging import getLogger

from .utils import get_optimizer
from .replay_memory import ReplayMemory


from .model.aThreeCFinal.model import ActorCritic
from .doom.utils import process_buffers
from torch.autograd import Variable
import torch.nn.functional as F

logger = getLogger()


'''
This will train A3C Algo
'''
class A3CTrainer(object):
	def __init__(self,rank, params, params2,shared_model, optimizer,game,eval_fn,evaluation_agent=False):
		self.rank = rank
		self.params = params
		self.params2 = params2
		self.shared_model = shared_model
		self.optimizer = optimizer
		self.game = game
		self.n_iter = 0
		self.eval_fn = eval_fn
		self.evaluation_agent=evaluation_agent

	def ensure_shared_grads(self,model, shared_model):
		for param, shared_param in zip(model.parameters(), shared_model.parameters()):
			if shared_param.grad is not None:
				return
			shared_param._grad = param.grad


	'''
	Of no use in a3c case because will take dump once every hr
	'''
	def dump_model(self,model,start_iter):
		model_name = 'periodic-%i.pth' % (self.n_iter - start_iter)
		model_path = os.path.join(self.params.dump_path, model_name)
		logger.info('Periodic dump: %s' % model_path)
		torch.save(model.state_dict(), model_path)


	def evaluate_model(self, start_iter,model):
		self.game.close()
		# if we are using a recurrent network, we need to reset the history
		new_score = self.eval_fn(model,self.game,self.params, self.n_iter)

		if new_score > self.best_score:
			self.best_score = new_score
			logger.info(str(self.rank)+'New best score: %f' % self.best_score)
			model_name = 'best-%i.pth' % (self.n_iter - start_iter)
			model_path = os.path.join(self.params.dump_path, model_name)
			logger.info('Best model dump: %s' % model_path)
			torch.save(model.state_dict(), model_path)
		model.train()
		self.start_game()


	def start_game(self):
		map_id = np.random.choice(self.params.map_ids_train)
		logger.info("Training on map %i ..." % map_id)
		self.game.start(map_id=map_id,
						episode_time=self.params.episode_time,
						log_events=False,
						manual_control=False)
		if hasattr(self.params, 'randomize_textures'):
			self.game.randomize_textures(self.params.randomize_textures)
		if hasattr(self.params, 'init_bots_health'):
			self.game.init_bots_health(self.params.init_bots_health)

	'''
	Game is the modified game object
	'''
	def run(self):
		self.start_game()
		env = self.game.game #Doom object
		torch.manual_seed(self.params2.seed + self.rank)
		env.set_seed(self.params2.seed + self.rank)

		dump_frequency = self.params.dump_freq
		start_iter = self.n_iter
		last_eval_iter = self.n_iter

		n_actions= self.game.action_builder.n_actions
		# height = self.params.height
		# width = self.params.width
		#num_inputs = (self.params.n_fm, height, width)
		#num_inputs = self.params.n_fm
		num_inputs = 3 #ToDO: See what is this i am thinking of number of channel in input image.
		model = ActorCritic(num_inputs,n_actions)

		state, game_features = process_buffers(self.game, self.params)#Get first screen

		#state = torch.from_numpy(screen)

		done = True  # when the game is done
		episode_length = 0  # initializing the length of an episode to 0

		while True:  # repeat
			self.n_iter += 1
			episode_length += 1  # incrementing the episode length by one
			model.load_state_dict(self.shared_model.state_dict())

			if done:  # if it is the first iteration of the while loop or if the game was just done, then:
				cx = Variable(torch.zeros(1, 256))  # the cell states of the LSTM are reinitialized to zero
				hx = Variable(torch.zeros(1, 256))  # the hidden states of the LSTM are reinitialized to zero
			else:  # else:
				cx = Variable(cx.data)  # we keep the old cell states, making sure they are in a torch variable
				hx = Variable(hx.data)  # we keep the old hidden states, making sure they are in a torch variable

			values = []  # initializing the list of values (V(S))
			log_probs = []  # initializing the list of log probabilities
			rewards = []  # initializing the list of rewards
			entropies = []  # initializing the list of entropies

			for step in range(self.params2.num_steps):  # going through the num_steps exploration steps
				value, action_values, (hx, cx) = model((Variable(torch.FloatTensor(state).unsqueeze(0)), (hx, cx)))
				prob = F.softmax(action_values)
				log_prob = F.log_softmax(action_values)
				entropy = -(log_prob * prob).sum(1)
				entropies.append(entropy)
				action = prob.multinomial(num_samples=1).data
				#print('vvvvvvvvvv',action.numpy(),type(action))
				log_prob = log_prob.gather(1,Variable(action))  # getting the log prob associated to this selected action
				values.append(value)  # storing the value V(S) of the state
				log_probs.append(log_prob)

				self.game.make_action(int(action.numpy()), self.params.frame_skip)
				state, game_features = process_buffers(self.game, self.params)  # Get first screen

				reward=self.game.reward
				done=self.game.is_final()

				#state, reward, done, _ = env.step(action.numpy())

				reward = max(min(reward, 1), -1)  # clamping the reward between -1 and +1
				if done:  # if the episode is done:
					episode_length = 0  # we restart the environment
					self.game.reset()
					screen, game_features = process_buffers(self.game, self.params)
					#state = env.reset()  # we restart the environment

				#state = torch.from_numpy(screen)
				rewards.append(reward)
				if done:  # if we are done
					break

			R = torch.zeros(1, 1)  # intializing the cumulative reward
			if not done:  # if we are not done:
				value, _, _ = model((Variable(torch.FloatTensor(state).unsqueeze(0)), (
				hx, cx)))  # we initialize the cumulative reward with the value of the last shared state
				R = value.data  # we initialize the cumulative reward with the value of the last shared state
			values.append(Variable(R))  # storing the value V(S) of the last reached state S
			policy_loss = 0  # initializing the policy loss
			value_loss = 0  # initializing the value loss
			R = Variable(R)  # making sure the cumulative reward R is a torch Variable
			gae = torch.zeros(1, 1)  # initializing the Generalized Advantage Estimation to 0
			for i in reversed(
					range(len(rewards))):  # starting from the last exploration step and going back in time
				R = self.params2.gamma * R + rewards[i]  # R = gamma*R + r_t = r_0 + gamma r_1 + gamma^2 * r_2 ... + gamma^(n-1)*r_(n-1) + gamma^nb_step * V(last_state)
				advantage = R - values[i]  # R is an estimator of Q at time t = i so advantage_i = Q_i - V(state_i) = R - value[i]
				value_loss = value_loss + 0.5 * advantage.pow(2)  # computing the value loss
				TD = rewards[i] + self.params2.gamma * values[i + 1].data - values[
					i].data  # computing the temporal difference
				gae = gae * self.params2.gamma * self.params2.tau + TD  # gae = sum_i (gamma*tau)^i * TD(i) with gae_i = gae_(i+1)*gamma*tau + (r_i + gamma*V(state_i+1) - V(state_i))
				policy_loss = policy_loss - log_probs[i] * Variable(gae) - 0.01 * entropies[i]  # computing the policy loss
			self.optimizer.zero_grad()  # initializing the optimizer
			(policy_loss + 0.5 * value_loss).backward()  # we give 2x more importance to the policy loss than the value loss because the policy loss is smaller

			torch.nn.utils.clip_grad_norm(model.parameters(),40)  # clamping the values of gradient between 0 and 40 to prevent the gradient from taking huge values and degenerating the algorithm
			self.ensure_shared_grads(model,self.shared_model)  # making sure the model of the agent and the shared model share the same gradient
			self.optimizer.step()  # running the optimization step


			# if (dump_frequency > 0 and (self.n_iter - last_dump_iter) % dump_frequency == 0):
			# 	self.dump_model(model,start_iter)
			# 	last_dump_iter = self.n_iter

			# evaluation
			if self.evaluation_agent:
				if (self.n_iter - last_eval_iter) % self.params.eval_freq == 0:
					model.load_state_dict(self.shared_model.state_dict())
					self.evaluate_model(model,start_iter)
					last_eval_iter = self.n_iter


		self.game.close()




	def __call__(self, *args, **kwargs):
		self.run(*args,**kwargs)




class Trainer(object):

	def __init__(self, params, game, network, eval_fn, parameter_server=None):
		optim_fn, optim_params = get_optimizer(params.optimizer)
		self.optimizer = optim_fn(network.module.parameters(), **optim_params)
		self.parameter_server = parameter_server
		self.params = params
		self.game = game
		self.network = network
		self.eval_fn = eval_fn
		self.state_dict = self.network.module.state_dict()
		self.n_iter = 0
		self.best_score = -1000000

	def start_game(self):
		map_id = np.random.choice(self.params.map_ids_train)
		logger.info("Training on map %i ..." % map_id)
		self.game.start(map_id=map_id,
						episode_time=self.params.episode_time,
						log_events=False,
						manual_control=False)
		if hasattr(self.params, 'randomize_textures'):
			self.game.randomize_textures(self.params.randomize_textures)
		if hasattr(self.params, 'init_bots_health'):
			self.game.init_bots_health(self.params.init_bots_health)

	def run(self):
		self.start_game()
		self.network.reset()

		network_type = self.params.network_type
		update_frequency = self.params.update_frequency
		log_frequency = self.params.log_frequency
		dump_frequency = self.params.dump_freq

		# log current training loss
		current_loss = self.network.new_loss_history()

		last_states = []
		start_iter = self.n_iter
		last_eval_iter = self.n_iter
		last_dump_iter = self.n_iter

		self.network.module.train()

		while True:
			self.n_iter += 1

			if self.game.is_final():
				self.game.reset()     # dead or end of episode
				self.network.reset()  # reset internal state (RNNs only)

			self.game.observe_state(self.params, last_states)

			# select the next action. `action` will correspond to an action ID
			# if we use non-continuous actions, otherwise it will correspond
			# to a set of continuous / discontinuous actions.
			# if DQN, epsilon greedy or action with the highest score
			random_action = network_type.startswith('dqn') and self.epsilon_greedy()
			if (network_type.startswith('dqn') and
					(not random_action or self.params.recurrence != '')):
				self.network.module.eval()
				action = self.network.next_action(last_states, save_graph=True)
				self.network.module.train()
			if random_action:
				action = np.random.randint(self.params.n_actions)

			# perform the action, and skip some frames
			self.game.make_action(action, self.params.frame_skip)

			# save last screens / features / action
			self.game_iter(last_states, action)

			# evaluation
			if (self.n_iter - last_eval_iter) % self.params.eval_freq == 0:
				self.evaluate_model(start_iter)
				last_eval_iter = self.n_iter

			# periodically dump the model
			if (dump_frequency > 0 and
					(self.n_iter - last_dump_iter) % dump_frequency == 0):
				self.dump_model(start_iter)
				last_dump_iter = self.n_iter

			# log current average loss
			if self.n_iter % (log_frequency * update_frequency) == 0:
				logger.info('=== Iteration %i' % self.n_iter)
				self.network.log_loss(current_loss)
				current_loss = self.network.new_loss_history()

			train_loss = self.training_step(current_loss)
			if train_loss is None:
				continue

			# backward
			self.optimizer.zero_grad()
			sum(train_loss).backward()
			for p in self.network.module.parameters():
				p.grad.data.clamp_(-5, 5)

			# update
			self.sync_update_parameters()

		self.game.close()

	def game_iter(self, last_states, action):
		raise NotImplementedError

	def training_step(current_loss):
		raise NotImplementedError

	def epsilon_greedy(self):
		"""
		For DQN models, return whether we randomly select the next action.
		"""
		start_decay = self.params.start_decay
		stop_decay = self.params.stop_decay
		final_decay = self.params.final_decay
		if final_decay == 1:
			return True
		slope = float(start_decay - self.n_iter) / (stop_decay - start_decay)
		p_random = np.clip((1 - final_decay) * slope + 1, final_decay, 1)
		return np.random.rand() < p_random

	def evaluate_model(self, start_iter):
		self.game.close()
		# if we are using a recurrent network, we need to reset the history
		new_score = self.eval_fn(self.game, self.network,
								 self.params, self.n_iter)
		if new_score > self.best_score:
			self.best_score = new_score
			logger.info('New best score: %f' % self.best_score)
			model_name = 'best-%i.pth' % (self.n_iter - start_iter)
			model_path = os.path.join(self.params.dump_path, model_name)
			logger.info('Best model dump: %s' % model_path)
			torch.save(self.network.module.state_dict(), model_path)
		self.network.module.train()
		self.start_game()

	def dump_model(self, start_iter):
		model_name = 'periodic-%i.pth' % (self.n_iter - start_iter)
		model_path = os.path.join(self.params.dump_path, model_name)
		logger.info('Periodic dump: %s' % model_path)
		torch.save(self.network.module.state_dict(), model_path)

	def sync_update_parameters(self):
		server = self.parameter_server
		if server is None or server.n_processes == 1:
			self.optimizer.step()
			return
		shared_dict = server.state_dict
		grad_scale = 1. / server.n_processes
		if server.rank == 0:
			# accumulate shared gradients into the local copy
			for k in self.state_dict:
				self.state_dict[k].grad.mul_(grad_scale).add_(shared_dict[k].grad)
			# do optimization
			self.optimizer.step()
			# copy updated parameters
			self.sync_dicts(self.state_dict, shared_dict)
			# zero shared gradients
			for v in shared_dict.values():
				v.grad.zero_()
		else:
			# accumulate gradients
			for k in shared_dict:
				shared_dict[k].grad.add_(grad_scale, self.state_dict[k].grad)
			# copy shared parameters
			self.sync_dicts(shared_dict, self.state_dict)

	def sync_dicts(self, src, dst, attr='data'):
		# TODO: use page-locked memory for parameter server
		for k in src:
			getattr(dst[k], attr).copy_(getattr(src[k], attr))


class ReplayMemoryTrainer(Trainer):

	def __init__(self, params, *args, **kwargs):
		super(ReplayMemoryTrainer, self).__init__(params, *args, **kwargs)

		# initialize the replay memory
		self.replay_memory = ReplayMemory(
			params.replay_memory_size,
			(params.n_fm, params.height, params.width),
			params.n_variables, params.n_features
		)

	def game_iter(self, last_states, action):
		# store the transition in the replay table
		self.replay_memory.add(
			screen=last_states[-1].screen,
			variables=last_states[-1].variables,
			features=last_states[-1].features,
			action=action,
			reward=self.game.reward,
			is_final=self.game.is_final()
		)

	def training_step(self, current_loss):
		# enforce update frequency
		if self.n_iter % self.params.update_frequency != 0:
			return
		# prime the training
		if self.replay_memory.size < self.params.batch_size:
			return

		# sample from replay memory and compute predictions and losses
		memory = self.replay_memory.get_batch(
			self.params.batch_size,
			self.params.hist_size + (0 if self.params.recurrence == ''
									 else self.params.n_rec_updates - 1)
		)
		return self.network.f_train(loss_history=current_loss, **memory)
