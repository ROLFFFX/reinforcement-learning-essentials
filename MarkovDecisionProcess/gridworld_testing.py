from gridworld import gridworld 

def test_gridworld(P, slip_prob=0.2):
    """
    Test the correctness of the gridworld transition dictionary P.

    Args:
    P (dict): Transition probability dictionary.
    slip_prob (float): Probability of slipping.

    Returns:
    None. Prints the results of each test.
    """
    num_states = 25
    actions = [0, 1, 2, 3]
    passed = True  # Track overall test result

    def check_special_state(state, expected_next_state, expected_reward):
        """Check transitions for special states 1 and 3."""
        for action in actions:
            transitions = P[state][action]
            if len(transitions) != 2 or transitions[1] != (1 - slip_prob, expected_next_state, expected_reward):
                print(f"❌ Special state {state} failed for action {action}: {transitions}")
                return False
        return True

    def check_off_grid(state, action):
        """Check if an off-grid action is handled correctly."""
        transitions = P[state][action]
        if len(transitions) != 2 or transitions[1] != (1 - slip_prob, state, -1):
            print(f"❌ Off-grid action failed for state {state}, action {action}: {transitions}")
            return False
        return True

    def check_normal_transition(state, action, expected_next_state):
        """Check normal transitions for non-boundary, non-special states."""
        transitions = P[state][action]
        if len(transitions) != 2 or transitions[1] != (1 - slip_prob, expected_next_state, 0):
            print(f"❌ Normal transition failed for state {state}, action {action}: {transitions}")
            return False
        return True

    # Test each state
    for state in range(num_states):
        for action in actions:
            # Case 1: Special states
            if state == 1:
                if not check_special_state(1, 21, 10):
                    passed = False
                break
            if state == 3:
                if not check_special_state(3, 13, 5):
                    passed = False
                break

            # Case 2: Off-grid actions
            if state in range(5) and action == 0:  # Top row, moving north
                if not check_off_grid(state, action):
                    passed = False
            elif state in range(20, 25) and action == 2:  # Bottom row, moving south
                if not check_off_grid(state, action):
                    passed = False
            elif state % 5 == 0 and action == 3:  # Left column, moving west
                if not check_off_grid(state, action):
                    passed = False
            elif (state + 1) % 5 == 0 and action == 1:  # Right column, moving east
                if not check_off_grid(state, action):
                    passed = False

            # Case 3: Normal transitions
            else:
                if action == 0:  # North
                    expected_next_state = state - 5
                elif action == 1:  # East
                    expected_next_state = state + 1
                elif action == 2:  # South
                    expected_next_state = state + 5
                elif action == 3:  # West
                    expected_next_state = state - 1
                if not check_normal_transition(state, action, expected_next_state):
                    passed = False

    if passed:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed. Check the output above.")


if __name__ == "__main__":
    P = gridworld(slip_prob=0.2)
    test_gridworld(P)