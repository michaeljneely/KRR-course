"""
    An algorithm to solve sequential planning problems (specified in a format unique to this assignment) with Clingo

    Author: Michael Neely
    Student ID: 12547190

    References:
        - plasp 3: Towards Effective ASP Planning (Dimopoulos et al. 2018)
            https://arxiv.org/pdf/1812.04491.pdf
        - Potassco plasp
            https://github.com/potassco/plasp/

    Solution Notes:
    --------------
    "A scholar knows not to waste time rediscovering information already known" - Brandon Sanderson, The Way of Kings

    I'm not re-inventing the wheel, just putting pieces together.

    The goal is to convert a pseudo-PDDL description to ASP facts and then solve the planning problem.
    A solution is found with a meta-encoding. "plasp 3: Towards Effective ASP Planning" (Dimopoulos et al. 2018 -
    https://arxiv.org/pdf/1812.04491.pdf) details sequential and parallel meta-encodings, the source code for which
    is available in the Plasp repository (https://github.com/potassco/plasp). The paper specifically refers to the
    strips-incremental.lp encoding, but the authors also include a simplified sequential meta-encoding in
    sequential-horizon.lp. This simplified encoding yields a plan as a sequence of time-stamped actions if and only
    if the plan's length matches the externally defined horizon constant.

    At a high-level, my solution:
    1. Adds, explains, and slightly alters a minimal version of plasp's sequential-horizon meta-encoding
    2. Converts the pseudo-PDDL description to ASP facts in a manner similar to the 'translate' feature of plasp
    3. Finds the optimal (shortest) plan by progressively incrementing the horizon from t = [1, t_max] (inclusive)
    4. Returns the optimal plan or None if no solution is found in the range [1, t_max]

    Example Input:
        initial: (At(C1, SFO) & At(C2, JFK) & At(P1, SFO) & At(P2, JFK) & Cargo(C1) & Cargo(C2) & Plane(P1) & Plane(P2)
                 & Airport(SFO) & Airport(JFK))
        goals: (At(C1, JFK) & At(C2, SFO))
        action: Load(c, p, a); (At(c, a) & At(p, a) & Cargo(c) & Plane(p) & Airport(a)); (Contains(c, p) & ~At(c, a))
        action: Unload(c, p, a); (Contains(c, p) & At(p, a) & Cargo(c) & Plane(p) & Airport(a)); (At(c, a)
                & ~Contains(c, p))
        action: Fly(p, f, to); (At(p, f) & Plane(p) & Airport(f) & Airport(to)); (At(p, to) & ~At(p, f))
        t_max: 10

    Example Output:
        ['Load(C2,P2,JFK)', 'Fly(P2,JFK,SFO)', 'Load(C1,P1,SFO)', 'Unload(C2,P2,SFO)', 'Fly(P1,SFO,JFK)',
        'Unload(C1,P1,JFK)']
"""

# Standard library
import collections
import itertools
import re
from typing import Dict, List, Tuple, Union

# External imports
import clingo

# Local imports
from planning import PlanningProblem, Action, Expr

###
### Helper functions
###
def init_sequential_planning_program() -> str:
    """Initialize an ASP sequential planning program with a minimized version of the sequential horizon meta-encoding
    as detailed in "plasp 3: Towards Effective ASP Planning" (Dimopoulos et al. 2018 -
    https://arxiv.org/pdf/1812.04491.pdf) and explicitly declared in the plasp repository:
    https://github.com/potassco/plasp/blob/master/encodings/sequential-horizon.lp

    All credit goes to the plasp authors. My only contribution is minor simplification and
    explanation of each group of statements in detail

    :return: Initialized ASP sequential planning program, ready to be extended with facts from a planning problem
    :rtype: str
    """
    # We reason about the state of the world at particular time steps: [0, t_max]
    seq_encoding = 'time(0..horizon).\n'

    # Predicates evaluate to True or False
    seq_encoding += 'boolean(true).\n'
    seq_encoding += 'boolean(false).\n'
    # The contains/2 atom captures this relationship
    seq_encoding += 'contains(X, value(X, B)) :- predicate(X), boolean(B).\n'

    # The initial state is at time t=0
    # The holds/3 atom captures the value of a predicate at a particular timestep t >= 0
    seq_encoding += 'holds(Predicate, Value, 0) :- initialState(Predicate, Value).\n'

    # Closed World Assumption (CWA): Any ground atoms in the initial state which are not explicitly declared True
    # are set to False
    seq_encoding += 'initialState(X, value(X, false)) :- predicate(X), not initialState(X, value(X, true)).\n'

    # The solution to the planning problem is extracted from occurs/2 atoms
    # This is a sequential encoding: only one action may occur at a particular timestep
    # Also, actions may only occur AFTER the initial state.
    seq_encoding += '1 {occurs(Action, T) : action(Action)} 1 :- time(T), T > 0.\n'

    # An action may not occur unless its preconditions are met (i.e., for an action to occur at time t,
    # all applicable predicates must hold the values specified in the precondition at time t-1)
    seq_encoding += (
        ':- occurs(Action, T), precondition(Action, Predicate, Value), '
        'not holds(Predicate, Value, T - 1).\n'
    )

    # Capture the effects of an action: at time t, the value of a predicate is changed to the one specified in the
    # action's effect as long as the action was valid (see previous statement).
    seq_encoding += (
        'caused(Predicate, Value, T) :- '
        'occurs(Action, T), '
        'effect(Action, Predicate, Value), '
        'holds(PredicatePre, ValuePre, T - 1) : precondition(Action, PredicatePre, ValuePre).\n'
    )

    # A predicate is considered modified if its value was changed by an action
    seq_encoding += 'modified(Predicate, T) :- caused(Predicate, Value, T).\n'

    # The so-called 'inertia' statements. At a particular timestep, the value of a predicate was either:
    # 1) Modified and therefore holds a new value
    seq_encoding += 'holds(Predicate, Value, T) :- caused(Predicate, Value, T).\n'
    # 2) Was not modified and therefore continues to hold its previous value
    seq_encoding += (
        'holds(predicate(V), Value, T) :- holds(predicate(V), Value, T - 1), '
        'not modified(predicate(V), T), time(T).\n'
    )

    # The goal is not met unless the appropriate predicates hold their goal values at the final timestep
    seq_encoding += ':- goal(Predicate, Value), not holds(Predicate, Value, horizon).\n'

    return seq_encoding


def make_positive(expression: Expr) -> Expr:
    """Make any expression positive by removing the ~ if needed

    :param expression: A potentially negative expression
    :type expression: Expr
    :return: A guaranteed positive expression
    :rtype: Expr
    """
    if expression.op == '~':
        new_expression = Expr(expression.args[0].op, *expression.args[0].args)
        return new_expression
    return expression


def is_variable(arg: Expr) -> bool:
    """Check if an expression argument is a variable. In the psuedo-PDDL description used in this assignment
     variables are lower case and constants/predicates are upper case

    :param arg: An expression argument
    :type arg: Expr
    :return: True if the argument is a variable, False otherwise
    :rtype: bool
    """
    return str(arg)[0].islower()


def extract_constants_and_predicates(planning_problem: PlanningProblem) -> Tuple[List[Expr],
                                                                                 List[Tuple[Expr, int]],
                                                                                 Dict[str, Expr]]:
    """Extract all unique constants and predicates from a planning problem

    :param planning_problem: A description of the initial state, action(s), and goal(s) of a planning problem
    :type planning_problem: PlanningProblem
    :return: Constants as a list of strings, predicates as a list of (name, number of arguments) tuples, a map of
             constants per predicate.
    :rtype: Tuple[List[Expr], List[Tuple[Expr, int]], Dict[str, Expr]]:
    """
    seen_predicates = set()
    seen_constants = set()
    constants_per_predicate = collections.defaultdict(list)

    initial_predicates = planning_problem.initial
    # Make all predicates positive so we can extract the name via predicate.op
    goal_predicates = list(map(make_positive, planning_problem.goals))
    precondition_predicates = list(map(make_positive, [p for a in planning_problem.actions for p in a.precond]))
    postcondition_predicates = list(map(make_positive, [e for a in planning_problem.actions for e in a.effect]))

    all_predicates = initial_predicates + goal_predicates + precondition_predicates + postcondition_predicates

    for predicate in all_predicates:
        if predicate.op not in seen_predicates and not is_variable(predicate.op):
            seen_predicates.add((predicate.op, len(predicate.args)))
        for arg in predicate.args:
            if arg not in seen_constants and not is_variable(arg):
                seen_constants.add(arg)
                constants_per_predicate[predicate.op].append(arg)

    return list(seen_constants), list(seen_predicates), constants_per_predicate


def action_to_asp_facts(action: Action) -> str:
    """Convert a planning problem Action into a collection of ASP facts

    :param action: An action schema with preconditions and effects
    :type action: Action
    :return: A collection of asp facts for the action, precondition, and effects
    :rtype: str
    """
    fact_string = ''

    # Keep track of arguments in order to correctly label constants
    arg_map = {}
    variable_counter = 1
    for arg in action.args:
        if not is_variable(arg):
            arg_map[str(arg)] = f'constant("{str(arg)}")'
        else:
            arg_map[str(arg)] = f'X{variable_counter}'
            variable_counter += 1

    # First declare the action as:
    #   action(action(("?", X1, ..., Xn))) :- constant(X1), ..., constant(Xn).
    if arg_map.values():
        action_signature = f'action(("{action.name}", {", ".join(arg_map.values())}))'
        action_constants = ', '.join([f'constant({k})' for k in arg_map.values()])
        action_constants = ' :- ' + action_constants
    else:
        action_signature = f'action(("{action.name}"))'
        action_constants = ''
    fact_string += f'action({action_signature}){action_constants}.\n'


    # Declare the preconditions as:
    #   precondition(action_signature, predicate((...)), value(...(predicate(()), true or false))
    #       :- action(action((...))).
    # And Effects as:
    #   effect(action_signature, predicate((...)), value(...(predicate(()), true or false)) :- action(action((...))).
    preconditions = [('precondition', p) for p in action.precond]
    effects = [('effect', e) for e in action.effect]

    for name, expression in preconditions + effects:
        positive_expression = make_positive(expression)
        # map variables to X1,...,Xn and map constants to constant("...")
        action_arg_map = {}
        for arg in positive_expression.args:
            if str(arg) in arg_map:
                action_arg_map[str(arg)] = arg_map[str(arg)]
            else:
                action_arg_map[str(arg)] = f'constant("{str(arg)}")'

        arguments = ", ".join(action_arg_map[str(arg)] for arg in positive_expression.args)

        if arguments:
            predicate = f'predicate(("{positive_expression.op}", {arguments}))'
        else:
            predicate = f'predicate(("{positive_expression.op}"))'

        boolean_value = "false" if expression != positive_expression else "true"
        value = f'value({predicate}, {boolean_value})'

        fact_string += f'{name}({action_signature},{predicate}, {value}) :- action({action_signature}).\n'

    return fact_string


def planning_problem_to_asp_facts(planning_problem: PlanningProblem) -> str:
    """Translate a pseudo-PDDL description to a collection of ASP facts per an approach similar to
       plasp translate (https://github.com/potassco/plasp)

    :param planning_problem: A description of the initial state, action(s), and goal(s) of a planning problem
    :type planning_problem: PlanningProblem
    :return: A single string containing the ASP fact translations
    :rtype: str
    """
    asp_facts = ''

    constants, predicates, constants_per_predicate = extract_constants_and_predicates(planning_problem)

    # Step 1: Declare constants as constant(constant("?")).
    for constant in constants:
        asp_facts += f'constant(constant("{constant}")).\n'

    # Step 2: Declare predicates and their arguments as:
    #         predicate(predicate("?", X1, ..., Xn)) :- constant(X1), ..., constant(Xn).
    for name, num_arguments in predicates:
        if num_arguments > 0:
            predicate_variables = ', '.join([f'X{i}' for i in range(1, num_arguments +1)])
            predicate_constants = ', '.join([f'constant(X{i})' for i in range(1, num_arguments +1)])
            asp_facts += f'predicate(predicate("{name}", {predicate_variables})) :- {predicate_constants}.\n'
        else:
            asp_facts += f'predicate(predicate("{name}")).\n'

    # Step 3: Extract and declare actions with their pre and post conditions
    for action in planning_problem.actions:
        asp_facts += action_to_asp_facts(action)

    # Step 4: Extract and declare initial state as:
    #         initialState(predicate(("?", constant("?1"), ..., constant("?n"))),
    #                      value(predicate(("?", constant("?1"), ..., constant("?n"))), true)).
    for predicate in planning_problem.initial:
        constants = ', '.join([f'constant("{str(arg)}")' for arg in predicate.args])
        initial_state_predicate = f'predicate(("{predicate.op}", {constants}))'
        # Ground atom in the initial state are exclusively positive
        initial_state_value = f'value(predicate(("{predicate.op}", {constants})), true)'
        asp_facts += f'initialState({initial_state_predicate}, {initial_state_value}).\n'

    # Step 4: Extract and declare goals as:
    #         goal(predicate(("?", constant("?1"), ..., constant("?n"))),
    #                      value(predicate(("?", constant("?1"), ..., constant("?n"))), true OR false)).
    # This is an extra-tricky part because the goals can contain variables
    for goal in planning_problem.goals:
        positive_goal = make_positive(goal)
        goal_args = []
        for arg in positive_goal.args:
            # If it's a variable, get all its possibilities
            if is_variable(arg):
                # Case 1: We have seen constants for this predicate before
                if positive_goal.op in constants_per_predicate:
                    possibilities = map(lambda x: f'constant("{x}")', constants_per_predicate[positive_goal.op])
                # Case 2: We have no information. Assume it could be any constant
                else:
                    possibilities = map(lambda x: f'constant("{x}")', sum(constants_per_predicate.values(), []))
                goal_args.append(possibilities)
            else:
                goal_args.append([f'constant("{arg}")'])

        # If there is more than one possible assignment for a goal, declare the goal as a disjunction of all possible
        # assignments. Question for TA: is this correct?
        all_goals = []
        boolean_value = "false" if positive_goal != goal else "true"
        for possible_assignment in itertools.product(*goal_args):
            variables = ', '.join(list(possible_assignment))
            if variables:
                goal_predicate = f'predicate(("{positive_goal.op}", {variables}))'
                goal_value = f'value(predicate(("{positive_goal.op}", {variables})), {boolean_value})'
            else:
                goal_predicate = f'predicate(("{positive_goal.op}"))'
                goal_value = f'value(predicate(("{positive_goal.op}")), {boolean_value})'
            all_goals.append(f'goal({goal_predicate}, {goal_value})')
        asp_facts += f'{"; ".join(all_goals)}.\n'

    return asp_facts


def plan_step_to_expr(atom: clingo.Symbol) -> str:
    """Converts a plan step 'occurs' atom (see above functions) to a string expression expected by the plan
       verification function of this assignment. E.g.:
       occurs(action(("Go",constant("L1"),constant("L3"))),1) --> Go(L1, L3)

    :param atom: An 'occurs' atom
    :type atom: clingo.Symbol
    :return: The string expression of the atom
    :rtype: str
    """
    # The predicate and its arguments are double-quoted. Simply extract them
    matches = re.findall(r'\"(.+?)\"', str(atom))
    predicate = matches[0]
    args = f'({",".join(matches[1:])})' if matches[1:] else ''
    return predicate + args


###
### Core Algorithm
###
def solve_planning_problem_using_ASP(planning_problem: PlanningProblem, t_max: int) -> Union[List[str], None]:
    """If there is a plan of length at most t_max that achieves the goals of a given planning problem,
    starting from the initial state in the planning problem, returns such a plan of minimal length.
    If no such plan exists of length at most t_max, returns None.

    :param planning_problem: Planning problem for which a shortest plan is to be found
    :type planning_problem: PlanningProblem
    :param t_max: The upper bound on the length of plans to consider
    :type t_max: int
    :return: A list of string representation of expressions that composes a shortest plan for planning_problem
             (if some plan of length at most t_max exists), and None otherwise.
    :rtype: List[str]
    """
    # Part 1: Add sequential horizon meta-encoding
    sequential_encoding = init_sequential_planning_program()

    # Part 2: Add ASP facts from the given planning_program
    facts = planning_problem_to_asp_facts(planning_problem)

    asp_planning_program = sequential_encoding + facts

    # If an answer set is found, build the solution by extracting the occurs/2 atoms, sorting them by timestep, and
    # converting them to string expressions
    solution = []

    def on_model(model):

        nonlocal solution
        steps = [atom for atom in model.symbols(atoms=True) if atom.name == 'occurs']
        # each occurs atom is a tuple of (action, timestep)
        sorted_steps = sorted(steps, key=lambda tup: tup.arguments[1])
        solution = list(map(plan_step_to_expr, sorted_steps))

    # Part 3: Find the shortest plan by checking for answer sets starting at t=1
    for horizon in range(1, t_max + 1):

        time_bounded_program = f'#const horizon={horizon}.\n{asp_planning_program}'''

        # Encode and solve
        control = clingo.Control()
        control.add("base", [], time_bounded_program)
        control.ground([("base", [])])
        control.configuration.solve.models = 1 # we only need one feasible solution
        answer = control.solve(on_model=on_model)

        # Part 4 option 1: An optimal solution was found
        if answer.satisfiable:
            return solution

    # Part 4 option 2: Alas, no solution with length <= t_max was found
    return None
