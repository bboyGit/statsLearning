
import numpy as np

class HMM(object):

    def __init__(self, iter_num: int, state_num: int, obs: np.ndarray):
        """
        :param iter_num:  Iteration times while fitting parameters
        :param state_num: The number of hidden variable state
        :param obs: 1-D array. Observation sequence
        """
        self.iter_num = iter_num
        self.obs = obs
        self.n = state_num
        self.m = len(np.unique(obs))
        init_trans = np.random.rand(state_num, state_num)
        self.init_trans = init_trans/init_trans.sum(axis=1).reshape([state_num, 1])
        init_emit = np.random.rand(state_num, self.m)
        self.init_emit = init_emit/init_emit.sum(axis=1).reshape([state_num, 1])
        init_prob = np.random.rand(state_num)
        self.init_prob = init_prob/init_prob.sum()
        self.state_space = np.arange(state_num)
        self.obs_space = np.unique(obs)

    def __calcu_obs_forward(self, trans, emit, prob):
        """
        Desc: Compute P(O|trans, emit, prob) by forward algorithm
        :return: T*N matrix
        """
        T = len(self.obs)                      # length of observation
        N = self.n                             # number of states
        alpha = np.empty(shape=[T, N])         # initialize forward probability
        for t in range(T):
            obs_t_idx = np.where(self.obs[t] == self.obs_space)[0][0]
            for i in range(N):
                if t == 0:
                    alpha[t, i] = prob[i] * emit[i, obs_t_idx]
                else:
                    alpha[t, i] = emit[i, obs_t_idx] * np.sum([alpha[t-1, j] * trans[j, i] for j in range(N)])
        p_obs_param = alpha[T-1, :].sum()
        result = {'alpha': alpha, 'p_obs_param': p_obs_param}
        return result

    def __calcu_obs_backward(self, trans, emit, prob):
        """
        Desc: Compute P(O|trans, emit, prob) by backward algorithm
        :return: T*N matrix
        """
        T = len(self.obs)                 # length of observation
        N = self.n                        # number of states
        beta = np.zeros(shape=[T, N])
        for t in range(T-1, -1, -1):
            obs_t_idx = np.where(self.obs[t] == self.obs_space)[0][0]
            if t == T - 1:
                beta[t, :] = 1
            else:
                obs_t1_idx = np.where(self.obs[t + 1] == self.obs_space)[0][0]
                for i in range(N):
                    beta[t, i] = np.sum(beta[t + 1, :] * emit[:, obs_t1_idx] * trans[i, :])
        p_obs_param = np.sum(beta[0, :] * emit[:, obs_t_idx] * prob)
        result = {'beta': beta, 'p_obs_param': p_obs_param}
        return result

    def calcu_obs_prob(self, trans, emit, prob, method):
        """
        Desc: Compute P(O|trans, emit, prob)
        :param method: Str in 'forward' or 'backward'
        :return: Int representing P(O|trans, emit, prob)
        """
        if method == 'forward':
            result = self.__calcu_obs_forward(trans, emit, prob)
        elif method == 'backward':
            result = self.__calcu_obs_backward(trans, emit, prob)
        else:
            raise Exception('Only support for forward or backward algorithm')
        return result

    def __calcu_obs_2var(self, trans, emit, alpha, beta):
        """
        Desc: Compute P(i(t) = q_i, i(t+1) = q_j|O, trans, emit, prob)
        :return: (T-1)*N matrix
        """
        T = len(self.obs)  # length of observation
        N = self.n  # number of states
        y = np.zeros(shape=[T-1, N, N])
        for t in range(T-1):
            obs_t1_idx = np.where(self.obs[t+1] == self.obs_space)[0][0]
            for i in range(N):
                for j in range(N):
                    y[t, i, j] = beta[t+1, j] * alpha[t, i] * trans[i, j] * emit[j, obs_t1_idx]
            y[t, :, :] = y[t, :, :]/y[t, :, :].sum()
        return y

    def fit(self):
        """
        Desc: Learning parameters of hidden markov chain by Baum-Welch algorithm
        :return: fitted parameters: trans, emit and prob.
        """
        T = len(self.obs)                                                           # length of observation
        N = self.n                                                                  # number of states
        M = self.m                                                                  # number of observation states
        # (1) Initialize parameters
        trans = self.init_trans.copy()
        emit = self.init_emit.copy()
        prob = self.init_prob.copy()
        # (2) update parameters by Baum-Welch algorithm
        for n in range(self.iter_num):
            # (2.1) calculate P(i(t) = q_i|O, trans, emit, prob) and P(i(t) = q_i, i(t+1) = q_j|O, trans, emit, prob)
            alpha_result = self.__calcu_obs_forward(trans, emit, prob)
            beta_result = self.__calcu_obs_backward(trans, emit, prob)
            alpha = alpha_result['alpha']                                           # forward probability matrix
            beta = beta_result['beta']                                              # backward probability matrix
            r = (alpha * beta) / np.sum(alpha * beta, axis=1).reshape(T, 1)         # P(i(t) = q_i|O, trans, emit, prob)
            y = self.__calcu_obs_2var(trans, emit, alpha, beta)       # P(i(t) = q_i, i(t+1) = q_j|O, trans, emit, prob)
            # (2.2) update prob
            prob = r[0, :]
            # (2.3) update trans and emit
            for i in range(N):
                for j in range(max(N, M)):
                    if j < N:
                        trans[i, j] = y[:, i, j].sum()/r[:-1, i].sum()
                    if j < M:
                        emit[i, j] = np.sum([r[t, i] * (self.obs[t] == self.obs_space[j]) for t in range(T)])/r[:, i].sum()
        # (3) tidy result
        result = {'trans': trans, 'emit': emit, 'prob': prob}
        return result

    def generate_obs(self, trans=None, emit=None, prob=None, length: int = 10):
        """
        Desc: generate observation sequence of a given hidden markov chain
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
            state_i = self.state_space[idx]
            I.append(state_i)
            # (2) determine the current observation
            p1 = np.random.rand()
            obs_idx = np.where(p1 >= emit[idx, :].cumsum())[0][-1]
            obs_i = self.obs_space[obs_idx]
            obs.append(obs_i)
        result = {'hide_var': I, 'obs': obs}
        return result

    def decode_hiden_var(self, trans, emit, prob, method='viterbi'):
        """
        Desc: Solve problem of I_hat = argmax P(I|O, trans, emit, prob) by Viterbi algorithm or similarity algorithm
        :return: Sequence I_hat.
        """
        T = len(self.obs)
        N = self.n
        if method == 'similarity':
            alpha_result = self.__calcu_obs_forward(trans, emit, prob)
            beta_result = self.__calcu_obs_backward(trans, emit)
            alpha = alpha_result['alpha']                                      # forward probability matrix
            beta = beta_result['beta']                                         # backward probability matrix
            r = (alpha * beta) / np.sum(alpha * beta, axis=1).reshape(T, 1)    # P(i(t) = q_i|O, trans, emit, prob)
            I_hat = np.array([r[t, :].argmax() for t in range(T)])
        elif method == 'viterbi':
            p_max_until = np.zeros([T, N])   # max P(i(t) = i, i1,...,i(t-1), o1,...ot|trans, emit, prob) by i1...i(t-1)
            # (1) Derive the max probability of P(O, I|trans, emit, prob) and the corresponding finally state
            for t in range(T):
                obs_t_idx = np.where(self.obs[t] == self.obs_space)[0][0]
                if t == 0:
                    p_max_until[0, :] = prob * emit[:, obs_t_idx]
                else:
                    for i in range(N):
                        p_max_until[t, i] = np.max([p_max_until[t-1, j] * trans[j, i] for j in range(N)]) * emit[i, obs_t_idx]
            p_max = p_max_until[T-1, :].max()
            i_T = p_max_until[T-1, :].argmax()                             # Tth state of optimal sequence
            I_hat = [i_T]
            # (2) Traceback from T to 1 to derive states corresponding to optimal probability
            for t in range(T-1, -1, -1):
                i_t_plus1 = I_hat[-1]
                i_t = np.argmax(p_max_until[t-1, :] * trans[:, i_t_plus1])
                I_hat.append(i_t)
            I_hat = np.array(I_hat)[::-1]
        else:
            raise Exception('method could only be viterbi or similarity')

        return I_hat

