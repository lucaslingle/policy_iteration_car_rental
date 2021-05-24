"""
Implements Policy Iteration for Jack's Car Rental example in
Sutton and Barto:

http://www.incompleteideas.net/sutton/book/first/4/node4.html
"""


import numpy as np
import math
from collections import Counter

# the value function. we use the coordinates of a 2D array to represent each state.
values = np.zeros(dtype=np.float32, shape=(21,21))

# the policy is deterministic, and we represent it via the net num cars to move from store0 to store1.
# as with the value function, we use the coordinates of a 2D array to represent each state.
policy = np.zeros(dtype=np.int32, shape=(21,21))

# actions
actions = np.linspace(start=-5, stop=5, num=11).astype(np.int32)

# value function improvement must be less than this amount for all states in a sweep before we stop iterating.
delta_thresh = 0.01

# problem-specific quantities influencing the reward function in each state.
poisson_lambda_request_store_0 = 3.0
poisson_lambda_request_store_1 = 4.0

poisson_lambda_dropoffs_store_0 = 3.0
poisson_lambda_dropoffs_store_1 = 2.0

mvmt_reward_multiple = -2.0
renting_reward_multiple = 10.0

gamma = 0.90


def get_truncated_posson_probs(lam, max_n):
    numers = lam ** np.arange(0, max_n+1)
    denoms = np.array(list(map(lambda i: math.factorial(i), np.arange(0, max_n+1))))
    probs = np.exp(-lam) * numers / denoms
    return probs


def compute_bellman_backup(state_idx_i, state_idx_j, a):
    cars_at_store_0 = state_idx_i
    cars_at_store_1 = state_idx_j
    cars_at_store_0 -= a
    cars_at_store_1 += a

    assert cars_at_store_0 >= 0 and cars_at_store_0 <= 20
    assert cars_at_store_1 >= 0 and cars_at_store_1 <= 20

    # transition to next day.
    # requests during the day - influences reward at current timestep
    probs_cars_requested_from_store_0 = get_truncated_posson_probs(poisson_lambda_request_store_0,
                                                                   cars_at_store_0)
    probs_cars_requested_from_store_1 = get_truncated_posson_probs(poisson_lambda_request_store_1,
                                                                   cars_at_store_1)

    expected_reward_for_store_0_given_state_and_policy = 0.0
    if len(probs_cars_requested_from_store_0) > 0:
        expected_reward_for_store_0_given_state_and_policy = renting_reward_multiple * np.sum(probs_cars_requested_from_store_0 * np.arange(0, cars_at_store_0 + 1))

    expected_reward_for_store_1_given_state_and_policy = 0.0
    if len(probs_cars_requested_from_store_1) > 0:
        expected_reward_for_store_1_given_state_and_policy = renting_reward_multiple * np.sum(probs_cars_requested_from_store_1 * np.arange(0, cars_at_store_1 + 1))

    expected_reward_given_state_and_policy = expected_reward_for_store_0_given_state_and_policy + \
                                             expected_reward_for_store_1_given_state_and_policy + \
                                             mvmt_reward_multiple * math.fabs(a)

    # the bellmann backup is an expectation,
    # and so we can apply linearity of expectations to compute the expected reward separately

    # now for the next state's value.
    probs_cars_left_at_store_0 = np.zeros(21, dtype=np.float32) # 0 thru 20.
    if len(probs_cars_requested_from_store_0) == 0:
        probs_cars_left_at_store_0[0] = 1.0
    else:
        probs_cars_left_at_store_0[-len(probs_cars_requested_from_store_0):] = probs_cars_requested_from_store_0[::-1]
        probs_cars_left_at_store_0[0] += 1.0 - np.sum(probs_cars_requested_from_store_0)

    probs_cars_left_at_store_1 = np.zeros(21, dtype=np.float32) # 0 thru 20.
    if len(probs_cars_requested_from_store_1) == 0:
        probs_cars_left_at_store_1[0] = 1.0
    else:
        probs_cars_left_at_store_1[-len(probs_cars_requested_from_store_1):] = probs_cars_requested_from_store_1[::-1]
        probs_cars_left_at_store_1[0] += 1.0 - np.sum(probs_cars_requested_from_store_1)

    # now the dropoffs during the day.
    # these dropped off cars will be made available the day after.
    # this does not influence the day's reward but influences next state (cars at end of this next day)
    probs_dropped_off_at_store_0 = get_truncated_posson_probs(poisson_lambda_dropoffs_store_0,
                                                              20 - len(probs_cars_left_at_store_0)) # fixme
    probs_dropped_off_at_store_1 = get_truncated_posson_probs(poisson_lambda_dropoffs_store_1,
                                                              20 - len(probs_cars_left_at_store_1)) # fixme

    # for each possible count before cars are dropped off,
    # the number of cars dropped off at that location is independent
    # so the total number of cars left at the end of the day depends on the sum of these two quantities
    # each possible event leading to the same number of cars left at the end of the day is an event that is mutually exclusive
    # with other events for the given store, so we can just sum the individual probabilities to obtain the overall probability.
    #
    store_0_outcomes = Counter()  # dictionary with implicitly zero-initted keys.
    for dropped_off in np.arange(0, len(probs_dropped_off_at_store_0)):
        for existing in np.arange(0, len(probs_cars_left_at_store_0)):
            summ = dropped_off + existing
            prob = probs_dropped_off_at_store_0[dropped_off] * probs_cars_left_at_store_0[existing]
            store_0_outcomes[summ] += prob

    store_1_outcomes = Counter()  # dictionary with implicitly zero-initted keys.
    for dropped_off in np.arange(0, len(probs_dropped_off_at_store_1)):
        for existing in np.arange(0, len(probs_cars_left_at_store_1)):
            summ = dropped_off + existing
            prob = probs_dropped_off_at_store_1[dropped_off] * probs_cars_left_at_store_1[existing]
            store_1_outcomes[summ] += prob

    expected_value_next_state = 0.0
    for hypothetical_store_0_nextstate in store_0_outcomes:
        for hypothetical_store_1_nextstate in store_1_outcomes:
            value_nextstate = values[hypothetical_store_0_nextstate, hypothetical_store_1_nextstate]
            prob = store_0_outcomes[hypothetical_store_0_nextstate] * store_1_outcomes[
                hypothetical_store_1_nextstate]
            expected_value_next_state += prob * value_nextstate

    vtarg = expected_reward_given_state_and_policy + gamma * expected_value_next_state
    return vtarg


def evaluate_policy():
    while True:
        policy_evaluations_so_far = 0
        print(f"policy_evaluations_so_far: {policy_evaluations_so_far}")
        stop_request = False

        state_ids = np.random.permutation(21*21)
        delta = 0
        for i in range(0, 21*21):
            # end of the day
            state_id = state_ids[i]
            state_idx_i, state_idx_j = state_id // 21, state_id % 21

            # over night
            a = policy[state_idx_i, state_idx_j]

            # compute expectation of what happens during the next day:
            # expected return based on one-step bellman backup.
            vtarg = compute_bellman_backup(state_idx_i, state_idx_j, a)
            vprev = values[state_idx_i, state_idx_j]
            values[state_idx_i, state_idx_j] = vtarg

            # delta_thresh
            delta = max(delta, math.fabs(vtarg - vprev))

            if delta < delta_thresh:
                stop_request = True
                break

        if stop_request:
            break

        policy_evaluations_so_far += 1
        #print(f"new_values: {policy}")


def improve_policy():
    for i in range(0, 21*21):
        state_id = i
        state_idx_i, state_idx_j = state_id // 21, state_id % 21
        argmax_a = None
        max_qpi_sa = None
        for a_idx in range(len(actions)):
            a = actions[a_idx]
            hypothetical_cars_at_store_0 = state_idx_i-a
            hypothetical_cars_at_store_1 = state_idx_j+a

            if hypothetical_cars_at_store_0 < 0 or hypothetical_cars_at_store_0 > 20:
                continue

            if hypothetical_cars_at_store_1 < 0 or hypothetical_cars_at_store_1 > 20:
                continue

            qpi_sa = compute_bellman_backup(state_idx_i, state_idx_j, a)

            if argmax_a is None:
                argmax_a = a
                max_qpi_sa = qpi_sa
            else:
                if qpi_sa >= max_qpi_sa:
                    argmax_a = a
                    max_qpi_sa = qpi_sa

        policy[state_idx_i, state_idx_j] = argmax_a



if __name__ == '__main__':
    for _ in range(10):
        evaluate_policy() # it works !!!
        improve_policy() # does this???

    print(values)
