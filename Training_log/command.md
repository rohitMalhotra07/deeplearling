# DeepLearning_P2
The second practice for Deep Learning course

### Deathmatch, RNN+lstm, gpu
python arnold.py --scenario deathmatch --wad deathmatch_rockets --n_bots 8 --action_combinations "move_fb;move_lr;turn_lr;attack" --frame_skip 4 --game_features "enemy" --network_type dqn_rnn --recurrence lstm --n_rec_updates 5 --gpu_id 0 --replay_memory_size 100000

### Deathmatch, RNN, gpu
python arnold.py --scenario deathmatch --wad deathmatch_rockets --n_bots 8 --action_combinations "move_fb;move_lr;turn_lr;attack" --frame_skip 4 --game_features "enemy" --network_type dqn_rnn --recurrence rnn --n_rec_updates 5 --gpu_id 0 --replay_memory_size 100000

### Deathmatch, Feedforward, gpu
python arnold.py --scenario deathmatch --wad deathmatch_rockets --n_bots 8 --action_combinations "move_fb;move_lr;turn_lr;attack" --frame_skip 4 --game_features "enemy" --network_type dqn_ff --gpu_id 0 --replay_memory_size 100000
