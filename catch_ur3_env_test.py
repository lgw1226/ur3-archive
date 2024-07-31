import numpy as np

from catch_ur3.envs.catch_ur3 import CatchUR3Env


def test(args):

    render_mode = args.render_mode
    num_episode = args.num_test_episode

    env = CatchUR3Env(render_mode=render_mode)

    for idx_episode in range(num_episode):

        obs, info = env.reset()

        for i in range(100):
            action = np.zeros(env.ur3_nqpos)
            obs, reward, terminated, truncated, info = env.step(action)

            if render_mode:
                frame = env.render()
    
    env.close()


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--num_test_episode', type=int, default=10, help='N of test episodes to run (default: 10)')
    parser.add_argument('--render_mode', type=str, choices=['human', 'rgb_array', 'depth'], default='human', help='Rendering mode (default: human)')

    args = parser.parse_args()

    test(args)