
import numpy as np

class HMM(object):

    def __init__(self, max_iter: int, thresh: float, init_trans: np.ndarray, init_emit: np.ndarray,
                 init_prob: np.ndarray, obs: np.ndarray):
        """
        :param max_iter:  maximum iteration times while fitting parameters
        :param thresh: threshold of breaking EM(Baum-Welch) algorithm loop
        :param init_trans: initial transfer probability matrix
        :param init_emit: initial emit probability matrix
        :param init_prob: initial state probability vector
        :param obs: Observation sequence
        """
        self.max_iter = max_iter
        self.thresh = thresh
        self.init_trans = init_trans
        self.init_emit = init_emit
        self.init_prob = init_prob
        self.obs = obs

    def fit(self):
        """
        Desc: Learning parameters of hidden markov chain by Baum-Welch algorithm
        :return: fitted parameters: trans, emit and prob.
        """
        return

    def calcu_obs_prob(self, trans, emit, prob, method):
        """
        Desc: Compute P(O|trans, emit, prob)
        :param method: Str in 'direct', 'forward' or 'backward'
        :return: Int representing P(O|trans, emit, prob)
        """
        return

    def generate_obs(self, state_space, obs_space, trans=None, emit=None, prob=None, length: int =10):
        """
        Desc: generate observation sequence of a given hidden markov chain
        :param state_space: A list or 1-d array contain all the possible states
        :param obs_space: A list or 1-d array contain all the possible observations
        :param length: length of generated sequence
        :return: A observation sequence
        Note: state_space 和 trans 以及 obs_space 和 emit 中变量的顺序要完全对应。
        """
        I = []
        obs = []
        for i in range(length):
            p = np.random.rand()
            # (1) determine the current state
            if i == 0:
                idx = np.where(p >= prob.cumsum())[0][-1]
            else:
                pre_state = I[i-1]
                idx = np.where(p >= trans[idx, :].cumsum())[0][-1]
            state_i = state_space[idx]
            I.append(state_i)
            # (2) determine the current observation
            p1 = np.random.rand()
            obs_idx = np.where(p1 >= emit[idx, :].cumsum())[0][-1]
            obs_i = obs_space[obs_idx]
            obs.append(obs_i)
        result = {'hide_var': I, 'obs': obs}
        return  result

    def decode_hiden_var(self):
        """
        Desc: Solve problem of I_hat = argmax P(I|O, trans, emit, prob) by Viterbi algorithm or similarity algorithm
        :return: Sequence I_hat.
        """
        return


if __name__ == "__main__":
    pass







