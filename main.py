from Environment import Environment

#
env = Environment()
env.load_models('garb.pt', 'rand_sheep.pt')
env.free_train(2000, 30000, 'garb.pt', 'rand_sheep.pt')
env.save_models('garb.pt', 'rand_sheep.pt')
# env.load_models()
# env.play()
