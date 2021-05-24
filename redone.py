import numpy as np
import math
from collections import Counter


class CarRental:
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

    def _intraday_state_transition(self, state, action):
        """Compute intermediate car counts 'state' after cars are moved overnight."""
        next_state = (state[0]-action, state[1]+action)
        return next_state

    def is_valid_store_action(self, store_state, store_action):
        """
        :param store_state: number of cars at store
        :param store_action: integer net number of cars of cars to move to the store.
        :return:
        """
        store_state_next = store_state + store_action
        if (0 <= store_state_next[0] <= self._max_cars_per_store):
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

    def compute_store_revenue(self, store_state, store_action, store_observed_requests):
        """
        :param store_state: tuple containing number of cars at store 0, store 1
        :param store_action: integer net number of cars to move to the store.
        :param store_observed_requests: number of requests at the store.
        :return: revenue
        """
        assert self.is_valid_store_action(store_state, store_action)

        num_cars_at_store = store_state + store_action
        revenue = self.renting_reward_multiple * min(store_observed_requests, num_cars_at_store)

        return revenue

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
        possible_cars_rented = np.maximum(
            num_cars_at_store * np.ones(self.max_cars_per_store+1),
            np.arange(0, self.max_cars_per_store+1) # [0,max_cars_per_store].
        ) # indexes over different hypothetical request amounts.

        possible_revenues = self.renting_reward_multiple * possible_cars_rented
        expected_revenue = np.sum(probs_store_observed_requests * possible_revenues)

        return expected_revenue

    def compute_probs_store0_observed_requests(self):
        ints = np.arange(0, self.max_cars_per_store+1)  # [0, max_cars_per_store].
        numerators = np.exp(-self.lam_request_store_0) * (self.lam_request_store_0 ** ints)
        denominators = np.array(list(map(lambda k: math.factorial(k), list(ints))))
        probs = numerators / denominators
        probs[-1] += (1.0 - np.sum(probs))
        return probs

    def compute_probs_store1_observed_requests(self):
        ints = np.arange(0, self.max_cars_per_store+1)  # [0, max_cars_per_store].
        numerators = np.exp(-self.lam_request_store_1) * (self.lam_request_store_1 ** ints)
        denominators = np.array(list(map(lambda k: math.factorial(k), list(ints))))
        probs = numerators / denominators
        probs[-1] += (1.0 - np.sum(probs))
        return probs

    def compute_expected_franchise_revenue(self, state, action):
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
        expected_revenue = self.compute_expected_franchise_revenue(state, action)
        cost = self.mvmt_cost_multiple * math.fabs(action)
        expected_profit = expected_revenue - cost
        return expected_profit

    def compute_probs_store0_observed_dropoffs(self):
        ints = np.arange(0, self.max_cars_per_store + 1)  # [0, max_cars_per_store].
        numerators = np.exp(-self.lam_dropoff_store_0) * (self.lam_dropoff_store_0 ** ints)
        denominators = np.array(list(map(lambda k: math.factorial(k), list(ints))))
        probs = numerators / denominators
        probs[-1] += (1.0 - np.sum(probs))
        return probs

    def compute_probs_store1_observed_dropoffs(self):
        ints = np.arange(0, self.max_cars_per_store + 1)  # [0, max_cars_per_store].
        numerators = np.exp(-self.lam_dropoff_store_1) * (self.lam_dropoff_store_1 ** ints)
        denominators = np.array(list(map(lambda k: math.factorial(k), list(ints))))
        probs = numerators / denominators
        probs[-1] += (1.0 - np.sum(probs))
        return probs

    def compute_probabilities_of_store0_states(self, store0_state, store0_action):
        probs_store0_observed_requests = self.compute_probs_store0_observed_requests()
        probs_store0_observed_dropoffs = self.compute_probs_store0_observed_dropoffs()

        num_cars_at_store0 = store0_state + store0_action  # includes cars moved overnight
        probabilities = Counter()
        for num_requests in range(0, self.max_cars_per_store):
            prob_a = probs_store0_observed_requests[num_requests]
            cars_left_at_store0 = max(0, num_cars_at_store0 - num_requests)
            for dropoffs in range(0, self.max_cars_per_store):
                # the bounds on this for-loop are ok since if observed requests is negative we simply dont give em a car,
                # and the new cars that come in arent available til the next day, so we never have less than zero and never have more than twenty.
                prob_b = probs_store0_observed_dropoffs[dropoffs]
                cars_for_next_day = min(20, cars_left_at_store0+dropoffs)
                probabilities[cars_for_next_day] += prob_a * prob_b

        return probabilities

    def compute_probabilities_of_store1_states(self, store1_state, store1_action):
        probs_store1_observed_requests = self.compute_probs_store1_observed_requests()
        probs_store1_observed_dropoffs = self.compute_probs_store1_observed_dropoffs()

        num_cars_at_store1 = store1_state + store1_action  # includes cars moved overnight
        probabilities = Counter()
        for num_requests in range(0, self.max_cars_per_store):
            prob_a = probs_store1_observed_requests[num_requests]
            cars_left_at_store1 = max(0, num_cars_at_store1 - num_requests)
            for dropoffs in range(0, self.max_cars_per_store):
                # the bounds on this for-loop are ok since if observed requests is negative we simply dont give em a car,
                # and the new cars that come in arent available til the next day, so we never have less than zero and never have more than twenty.
                prob_b = probs_store1_observed_dropoffs[dropoffs]
                cars_for_next_day = min(20, cars_left_at_store1 + dropoffs)
                probabilities[cars_for_next_day] += prob_a * prob_b

        return probabilities

    def compute_expected_value_of_next_state(self, state, action, values):
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

    def compute_vtargets_for_state(self, state, action, values):
        E_r_t = self.compute_expected_franchise_profit(state, action)
        E_V_tp1 = self.compute_expected_value_of_next_state(state, action, values)
        return E_r_t + E_V_tp1

