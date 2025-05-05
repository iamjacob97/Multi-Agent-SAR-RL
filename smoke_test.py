def run_episode(env, agents, verbose=False):
    obs = env.reset()
    done = False
    steps = 0
    while not done:
        actions = [ag.act(o) for ag, o in zip(agents, obs)]
        obs, rewards, done, info = env.step(actions, agents)
        for i, ag in enumerate(agents):
            ag.update(obs[i], rewards[i], done, info)
        steps += 1
        # if verbose: env.render()
    return steps, info

