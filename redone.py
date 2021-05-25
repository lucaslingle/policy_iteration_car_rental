"""
Implements Policy Iteration for Jack's Car Rental example in
Sutton and Barto:

http://www.incompleteideas.net/sutton/book/first/4/node4.html
"""


import numpy as np
import math
from collections import Counter
import copy
import matplotlib.pyplot as plt


class JacksCarRental:
    def __init__(self, max_cars_per_store):
        self._max_cars_per_store = max_cars_per_store
        self._action_space = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

        self.lam_request_store_0 = 3.0
        self.lam_request_store_1 = 4.0
        self.lam_dropoff_store_0 = 3.0
        self.lam_dropoff_store_1 = 2.0

        self.mvmt_cost_multiple = 2.0
        self.renting_reward_multiple = 10.0

    @property
    def max_cars_per_store(self):
        return self._max_cars_per_store

    @property
    def action_space(self):
        return self._action_space

    def is_valid_store_action(self, store_state, store_action):
        """
        :param store_state: integer number of cars at store
        :param store_action: integer net number of cars of cars to move to the store.
        :return:
        """
        store_state_next = store_state + store_action
        if 0 <= store_state_next <= self._max_cars_per_store:
            return True
        else:
            return False

    def is_valid_action(self, state, action):
        """
        :param state: tuple containing number of cars at store 0, store 1
        :param action: integer net number of cars of cars to move from store 0 to store 1.
        :return:
        """
        store0_action_valid = self.is_valid_store_action(store_state=state[0], store_action=-action)
        store1_action_valid = self.is_valid_store_action(store_state=state[1], store_action=action)
        return store0_action_valid and store1_action_valid

    def get_valid_actions(self, state):
        """
        :param state: tuple containing number of cars at store 0, store 1
        :param state: integer net number of cars of cars to move from store 0 to store 1.
        :return:
        """
        valid_actions = []
        for action in self.action_space:
            if self.is_valid_action(state, action):
                valid_actions.append(action)
        return valid_actions

    def compute_expected_store_revenue(self, store_state, store_action, probs_store_observed_requests):
        """
        :param store_state: tuple containing number of cars at store 0, store 1
        :param store_action: integer net number of cars to move to the store.
        :param probs_store_observed_requests: array of length max_cars_per_store+2, 
            containing probabilities for [0, max_cars_per_store]. 
            the probabilities for anything beyond the max number of *possible* cars should be added on to final prob
            before passing to this function. 
        :return: expected revenue from store
        """
        assert self.is_valid_store_action(store_state, store_action)

        num_cars_at_store = store_state + store_action
        possible_cars_rented = np.minimum(
            num_cars_at_store * np.ones(self.max_cars_per_store+1),
            np.arange(0, self.max_cars_per_store+1) # [0,max_cars_per_store].
        ) # indexes over different hypothetical request amounts.

        possible_revenues = self.renting_reward_multiple * possible_cars_rented
        expected_revenue = np.sum(probs_store_observed_requests * possible_revenues)

        return expected_revenue

    def compute_probs_store0_observed_requests(self):
        """
        Computes the probability distribution over different number of cars being requested
        from store 0. The distribution is represented using an array of length max_cars_per_store+1,
        corresponding to the integers from 0 to max_cars_per_store.

        The probability that more than max_cars_per_store are requested shall be added on to the
        probability that max_cars_per_store are requested. This works because the two use-cases
        for this function are for computing expected reward and computing probability distributions
        over next states: the reward for extra requests beyond the maximum number of cars available is zero,
        and extra requests have no effect on state transition dynamics.

        :return: the probabilities as a numpy array.
        """
        ints = np.arange(0, self.max_cars_per_store+1)  # [0, max_cars_per_store].
        numerators = np.exp(-self.lam_request_store_0) * (self.lam_request_store_0 ** ints)
        denominators = np.array(list(map(lambda k: math.factorial(k), list(ints))))
        probs = numerators / denominators
        probs[-1] += (1.0 - np.sum(probs))
        return probs

    def compute_probs_store1_observed_requests(self):
        """
        Computes the probability distribution over different number of cars being requested
        from store 1. The distribution is represented using an array of length max_cars_per_store+1,
        corresponding to the integers from 0 to max_cars_per_store.

        The probability that more than max_cars_per_store are requested shall be added on to the
        probability that max_cars_per_store are requested. This works because the two use-cases
        for this function are for computing expected reward and computing probability distributions
        over next states: the reward for extra requests beyond the maximum number of cars available is zero,
        and extra requests have no effect on state transition dynamics.

        :return: the probabilities as a numpy array.
        """
        ints = np.arange(0, self.max_cars_per_store+1)  # [0, max_cars_per_store].
        numerators = np.exp(-self.lam_request_store_1) * (self.lam_request_store_1 ** ints)
        denominators = np.array(list(map(lambda k: math.factorial(k), list(ints))))
        probs = numerators / denominators
        probs[-1] += (1.0 - np.sum(probs))
        return probs

    def compute_expected_franchise_revenue(self, state, action):
        """
        Computes the expected revenue at the next timestep from car rentals at both stores 0 and 1.
        The revenues from the locations are independent, given the action of moving cars overnight.
        This is because the requests at each store follow independent poisson distributions.

        :param state: tuple containing number of cars at store 0, store 1
        :param action: integer net number of cars of cars to move from store 0 to store 1.
        :return: expected franchise revenue for the next timestep,
                 given the state and action at this timestep.
        """
        probs_store0_observed_requests = self.compute_probs_store0_observed_requests()
        probs_store1_observed_requests = self.compute_probs_store1_observed_requests()

        # the revenue streams are independent since cars arrive at both locations
        # according to independent poisson distributions.
        expected_store_0_revenue = self.compute_expected_store_revenue(
            store_state=state[0], store_action=-action,
            probs_store_observed_requests=probs_store0_observed_requests)

        expected_store_1_revenue = self.compute_expected_store_revenue(
            store_state=state[1], store_action=action,
            probs_store_observed_requests=probs_store1_observed_requests)

        expected_revenue = expected_store_0_revenue + expected_store_1_revenue
        return expected_revenue

    def compute_expected_franchise_profit(self, state, action):
        """
        Compute the expected profit at the next timestep, obtained from the expected franchise revenue,
        minus the expected cost from taking the action of moving a certain number of cars
        between stores overnight.

        :param state: tuple containing number of cars at store 0, store 1
        :param action: integer net number of cars of cars to move from store 0 to store 1.
        :return: expected franchise profit for the next timestep,
                 given the state and action at this timestep.
        """
        expected_revenue = self.compute_expected_franchise_revenue(state, action)
        cost = self.mvmt_cost_multiple * math.fabs(action)
        expected_profit = expected_revenue - cost
        return expected_profit

    def compute_probs_store0_observed_dropoffs(self):
        """
        Computes the probability distribution over different number of cars being requested
        from store 0. The distribution is represented using an array of length max_cars_per_store+1,
        corresponding to the integers from 0 to max_cars_per_store.

        The probability that more than max_cars_per_store are dropped off shall be added on to the
        probability that max_cars_per_store are dropped off. This works because the one use-case
        for this function is for computing the probability distribution over the next state:
        extra dropoffs have no effect on state transition dynamics, since extra cars beyond max_cars_per_store
        shall be sent back to the nationwide company, and removed from the problem.

        :return: the probabilities as a numpy array
        """
        ints = np.arange(0, self.max_cars_per_store + 1)  # [0, max_cars_per_store].
        numerators = np.exp(-self.lam_dropoff_store_0) * (self.lam_dropoff_store_0 ** ints)
        denominators = np.array(list(map(lambda k: math.factorial(k), list(ints))))
        probs = numerators / denominators
        probs[-1] += (1.0 - np.sum(probs))
        return probs

    def compute_probs_store1_observed_dropoffs(self):
        """
        Computes the probability distribution over different number of cars being requested
        from store 1. The distribution is represented using an array of length max_cars_per_store+1,
        corresponding to the integers from 0 to max_cars_per_store.

        The probability that more than max_cars_per_store are dropped off shall be added on to the
        probability that max_cars_per_store are dropped off. This works because the one use-case
        for this function is for computing the probability distribution over the next state:
        extra dropoffs have no effect on state transition dynamics, since extra cars beyond max_cars_per_store
        shall be sent back to the nationwide company, and removed from the problem.

        :return: the probabilities as a numpy array
        """
        ints = np.arange(0, self.max_cars_per_store + 1)  # [0, max_cars_per_store].
        numerators = np.exp(-self.lam_dropoff_store_1) * (self.lam_dropoff_store_1 ** ints)
        denominators = np.array(list(map(lambda k: math.factorial(k), list(ints))))
        probs = numerators / denominators
        probs[-1] += (1.0 - np.sum(probs))
        return probs

    def compute_probabilities_of_store0_states(self, store0_state, store0_action):
        """
        Computes the probability distribution over next states for store 0.
        This can be done independently of the computation for store 1,
        since the state transition dynamics depend only on the store0_state, store0_action
        and the two independent poisson processes for store 0 (requests and dropoffs).

        :param store0_state: integer number of cars at store 0.
        :param store0_action: integer net number of cars of cars to move to store 0.
        :return: the probabilities as a collections.Counter object.
        """
        probs_store0_observed_requests = self.compute_probs_store0_observed_requests()
        probs_store0_observed_dropoffs = self.compute_probs_store0_observed_dropoffs()

        num_cars_at_store0 = store0_state + store0_action  # includes cars moved overnight
        probabilities = Counter()
        for num_requests in range(0, self.max_cars_per_store):
            prob_a = probs_store0_observed_requests[num_requests]
            cars_left_at_store0 = max(0, num_cars_at_store0 - num_requests)
            for dropoffs in range(0, self.max_cars_per_store):
                prob_b = probs_store0_observed_dropoffs[dropoffs]
                cars_at_end_of_day_store0 = min(20, cars_left_at_store0+dropoffs)
                probabilities[cars_at_end_of_day_store0] += prob_a * prob_b

        return probabilities

    def compute_probabilities_of_store1_states(self, store1_state, store1_action):
        """
        Computes the probability distribution over next states for store 1.
        This can be done independently of the computation for store 0,
        since the state transition dynamics depend only on the store1_state, store1_action
        and the two independent poisson processes for store 1 (requests and dropoffs).

        :param store1_state: integer number of cars at store 1
        :param store1_action: integer net number of cars of cars to move to store 0.
        :return: the probabilities as a collections.Counter object.
        """
        probs_store1_observed_requests = self.compute_probs_store1_observed_requests()
        probs_store1_observed_dropoffs = self.compute_probs_store1_observed_dropoffs()

        num_cars_at_store1 = store1_state + store1_action  # includes cars moved overnight
        probabilities = Counter()
        for num_requests in range(0, self.max_cars_per_store):
            prob_a = probs_store1_observed_requests[num_requests]
            cars_left_at_store1 = max(0, num_cars_at_store1 - num_requests)
            for dropoffs in range(0, self.max_cars_per_store):
                prob_b = probs_store1_observed_dropoffs[dropoffs]
                cars_at_end_of_day_store1 = min(20, cars_left_at_store1 + dropoffs)
                probabilities[cars_at_end_of_day_store1] += prob_a * prob_b

        return probabilities

    def compute_expected_value_of_next_state(self, state, action, values):
        """
        Computes the expected value estimate of the next state, given the current state and action,
        and an array of value estimates.

        Our only use-case for this function is in the
        Bellman expectation equation backup, and for that purpose it suffices to
        compute the expectation over rewards and next values separately,
        thanks to the linearity of expectations.

        We further simplify the computation by computing the probabilities of state transitions
        for each store separately, since the store state transitions are independent,
        given their current states and our action.

        :param state: tuple containing number of cars at store 0, store 1
        :param action: integer net number of cars of cars to move from store 0 to store 1.
        :param values: 2D numpy array containing the value estimates for each state.
        :return: float expected value estimate.
        """
        store0_state = state[0]
        store0_action = -action
        store0_state_probs = self.compute_probabilities_of_store0_states(store0_state, store0_action)

        store1_state = state[1]
        store1_action = action
        store1_state_probs = self.compute_probabilities_of_store1_states(store1_state, store1_action)

        expected_value = 0.0
        for next_store0_state in store0_state_probs:
            prob_a = store0_state_probs[next_store0_state]
            for next_store1_state in store1_state_probs:
                prob_b = store1_state_probs[next_store1_state]
                expected_value += prob_a * prob_b * values[next_store0_state, next_store1_state]

        return expected_value

    def compute_bellman_backup(self, state, action, values, gamma):
        """
        Compute the value targets for the Bellman expectation equation.

        :param state: tuple containing number of cars at store 0, store 1
        :param action: integer net number of cars of cars to move from store 0 to store 1.
        :param values: 2D numpy array containing the value estimates for each state.
        :param gamma: discount factor.
        :return:
        """
        E_r_t = self.compute_expected_franchise_profit(state, action)
        E_V_tp1 = self.compute_expected_value_of_next_state(state, action, values)
        return E_r_t + gamma * E_V_tp1


class Agent:
    def __init__(self, max_cars_per_store, gamma):
        self._gamma = gamma
        self._max_cars_per_store = max_cars_per_store
        self._value_estimates = np.zeros(
            dtype=np.float32, shape=(max_cars_per_store+1, max_cars_per_store+1)
        )  # [0,max_cars_per_store]^2.
        self._policy = np.zeros(
            dtype=np.int32, shape=(max_cars_per_store+1, max_cars_per_store+1)
        )  # [0,max_cars_per_store]^2.

    @property
    def gamma(self):
        return self._gamma

    @property
    def max_cars_per_store(self):
        return self._max_cars_per_store

    @property
    def policy(self):
        return copy.deepcopy(self._policy)

    @property
    def value_estimates(self):
        return copy.deepcopy(self._value_estimates)

    def assign_new_values(self, new_value_estimates):
        self._value_estimates = new_value_estimates

    def assign_new_policy(self, new_policy):
        self._policy = new_policy


class Runner:
    def __init__(self):
        self.max_cars_per_store = 20
        self.gamma = 0.90
        self.delta_thresh = 0.001

        self.env = JacksCarRental(self.max_cars_per_store)
        self.agent = Agent(self.max_cars_per_store, self.gamma)

    def policy_evaluation(self):
        stop_request = False
        policy_evaluations_so_far = 0
        while True:
            print(f"policy_evaluations_so_far: {policy_evaluations_so_far}\n\n")

            state_ids = np.random.permutation((self.max_cars_per_store+1) * (self.max_cars_per_store+1))
            delta = 0
            for i in range(0, 21 * 21):
                # end of the day
                state_id = state_ids[i]
                state_idx_i, state_idx_j = state_id // 21, state_id % 21

                # over night
                action = self.agent.policy[state_idx_i, state_idx_j]

                # compute expectation of what happens during the next day:
                # expected return based on one-step bellman backup.
                state = (state_idx_i, state_idx_j)
                #print(f"policy_evaluation state {state}")
                values = self.agent.value_estimates
                vtarg = self.env.compute_bellman_backup(state, action, values, self.gamma)

                vprev = values[state_idx_i, state_idx_j]
                values[state_idx_i, state_idx_j] = vtarg

                self.agent.assign_new_values(values)

                delta = max(delta, math.fabs(vtarg - vprev))
                if delta < self.delta_thresh:
                    stop_request = True
                    print("stop requested")
                    break

            policy_evaluations_so_far += 1
            if stop_request:
                break

    def policy_improvement(self):
        print("policy_improvement time!")
        for i in range(0, self.max_cars_per_store * self.max_cars_per_store):
            state_id = i
            state_idx_i, state_idx_j = state_id // 21, state_id % 21
            state = (state_idx_i, state_idx_j)
            #print(f"policy_improvement state {state}")
            argmax_a = None
            max_qpi_sa = None

            for action in self.env.get_valid_actions(state):
                qpi_sa = self.env.compute_bellman_backup(state, action, self.agent.value_estimates, self.gamma)
                if argmax_a is None:
                    argmax_a = action
                    max_qpi_sa = qpi_sa
                else:
                    if qpi_sa >= max_qpi_sa:
                        argmax_a = action
                        max_qpi_sa = qpi_sa

            policy = self.agent.policy
            policy[state_idx_i, state_idx_j] = argmax_a
            self.agent.assign_new_policy(policy)

    def iterate(self):
        self.policy_evaluation()
        self.policy_improvement()


if __name__ == '__main__':
    runner = Runner()
    for itr in range(0, 4):
        print(itr)
        runner.iterate()

    print(runner.agent.value_estimates)
    print(runner.agent.value_estimates[0, 0])
    print(runner.agent.value_estimates[20,20])
    plt.imshow(runner.agent.value_estimates)
    plt.show()
