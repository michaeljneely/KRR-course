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

    The goal is to convert a pseudo-PDDL description to ASP facts and then solve the planning problem.
    A solution is found with a meta-encoding. "plasp 3: Towards Effective ASP Planning" (Dimopoulos et al. 2018 -
    https://arxiv.org/pdf/1812.04491.pdf) details sequential and parallel meta-encodings, the source code for which
    is available in the Plasp repository (https://github.com/potassco/plasp). The paper specifically refers to the
    strips-incremental.lp encoding, but the authors also include a simplified sequential meta-encoding in
    sequential-horizon.lp. This simplified encoding yields a plan as a sequence of time-stamped actions if and only
    if the plan's length matches the externally defined horizon constant.

    At a high-level, my solution:
    1. Adds (and explains) a minimal version of plasp's sequential-horizon meta-encoding
    2. Converts the pseudo-PDDL description to ASP facts in a manner similar to the 'translate' feature of plasp
    3. Finds the optimal (shortest) plan by progressively incrementing the horizon from t = [1, t_max] (inclusive)
    4. Returns the optimal plan or None if no solution is found in the range [1, t_max]
"""

# Standard library
import collections
import itertools
import re
from typing import List, Tuple, Union

# Local imports
from planning import PlanningProblem, Action, Expr, expr

# External imports
import clingo

###
### Helper functions
###
def init_sequential_planning_program() -> str:
    """Initialize an ASP sequential planning program with a minimized version of the sequential horizon meta-encoding
       as detailed in "plasp 3: Towards Effective ASP Planning" (Dimopoulos et al. 2018 -
       https://arxiv.org/pdf/1812.04491.pdf) and explicitly declared in the plasp repository:
       https://github.com/potassco/plasp/blob/master/encodings/sequential-horizon.lp

       All credit goes to the plasp authors. My only contribution is minor simplification and 
       explaining each group of statements in detail

    :return: Initialized ASP sequential planning program, ready to be extended with facts from a planning problem
    :rtype: str
    """
    # We reason about the state of the world at particular time steps: [0, t_max]
    asp_planning_program = 'time(0..horizon).\n'

    # Predicates contain values, which may be True or False
    asp_planning_program += 'boolean(true).\n'
    asp_planning_program += 'boolean(false).\n'
    # The contains/2 atom captures this relationship
    asp_planning_program += 'contains(X, value(X, B)) :- predicate(X), boolean(B).\n'

    # The initial state is at time t=0
    # The holds/3 atom captures the value of a predicate at a particular timestep t >= 0
    asp_planning_program += 'holds(Predicate, Value, 0) :- initialState(Predicate, Value).\n'

    # Closed World Assumption (CWA): Any ground atoms in the initial state which are not explicitly declared True
    # are set to False
    asp_planning_program += 'initialState(X, value(X, false)) :- predicate(X), not initialState(X, value(X, true)).\n'

    # The solution to the planning problem is extracted from occurs/2 atoms
    # This is a sequential encoding: only one action may occur at a particular timestep
    # Also, actions may only occur AFTER the initial state.
    asp_planning_program += '1 {occurs(Action, T) : action(Action)} 1 :- time(T), T > 0.\n'

    # An action may not occur unless its preconditions are met (i.e., for an action to occur at time t,
    # all applicable predicates must hold the values specified in the precondition at time t-1)
    asp_planning_program += ':- occurs(Action, T), precondition(Action, Predicate, Value), not holds(Predicate, Value, T - 1).\n'

    # Capture the effects of an action: at time t, the value of a predicate is changed to the one specified in the
    # action's effect as long as the action was valid (see previous statement).
    asp_planning_program += '''
        caused(Predicate, Value, T) :-
            occurs(Action, T),
            effect(Action, Predicate, Value),
            holds(PredicatePre, ValuePre, T - 1) : precondition(Action, PredicatePre, ValuePre).\n'''

    # A predicate is considered modified if its value was changed by an action
    asp_planning_program += 'modified(Predicate, T) :- caused(Predicate, Value, T).\n'

    # The so-called 'inertia' statements. At a particular timestep, the value of a predicate was either:
    # 1) Modified and therefore holds a new value
    asp_planning_program += 'holds(Predicate, Value, T) :- caused(Predicate, Value, T).\n'
    # 2) Was not modified and therefore continues to hold its previous value
    asp_planning_program += 'holds(predicate(V), Value, T) :- holds(predicate(V), Value, T - 1), not modified(predicate(V), T), time(T).\n'

    # The goal is not met unless the appropriate predicates hold their goal values at the final timestep
    asp_planning_program += ':- goal(Predicate, Value), not holds(Predicate, Value, horizon).\n'

    return asp_planning_program


def make_positive(expression: Expr) -> Expr:
    """Make any negated expression positive by removing the ~ if needed

    :param expression: A potentially negative expression
    :type expression: Expr
    :return: A guaranteed positive expression
    :rtype: Expr
    """
    if expression.op == '~':
        new_expression = Expr(expression.args[0].op, *expression.args[0].args)
        return new_expression
    return expression


def extract_constants_and_predicates(planning_problem: PlanningProblem) -> Tuple[List[str], List[Tuple[str, int]]]:
    """Extract all unique constants and predicates from a planning problem

    :param planning_problem: A description of the initial state, action(s), and goal(s) of a planning problem
    :type planning_problem: PlanningProblem
    :return: Constants as a list of strings and predicates as a list of (name, number of arguments) tuples
    :rtype: Tuple[List[str], List[Tuple(str, int)]]
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
        # Note: in the psuedo-PDDL description used in this assignment variables are lower case
        # and constants/predicates uppercase
        if predicate.op not in seen_predicates and predicate.op[0].isupper():
            seen_predicates.add((predicate.op, len(predicate.args)))
        for arg in predicate.args:
            if arg not in seen_constants and str(arg)[0].isupper():
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
        if str(arg)[0].isupper():
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
    #   precondition(action_signature, predicate((...)), value(...(predicate(()), true or false)) :- action(action((...))).
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
        predicate = f'predicate(("{positive_expression.op}", {arguments}))'
        boolean_value = "false" if expression != positive_expression else "true"
        value = f'value(predicate(("{positive_expression.op}", {arguments})), {boolean_value})'

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
        constants  = ', '.join([f'constant("{str(arg)}")' for arg in predicate.args])
        initial_state_predicate = f'predicate(("{predicate.op}", {constants}))'
        # Ground atom in the initial state are exclusively positive
        initial_state_value = f'value(predicate(("{predicate.op}", {constants})), true)'
        asp_facts += f'initialState({initial_state_predicate}, {initial_state_value}).\n'

    # Step 4: Extract and declare the goal as:
    #         goal(predicate(("?", constant("?1"), ..., constant("?n"))),
    #                      value(predicate(("?", constant("?1"), ..., constant("?n"))), true OR false)).
    for goal in planning_problem.goals:
        positive_goal = make_positive(goal)
        variables = [str(arg) for arg in positive_goal.args if str(arg)[0].islower()]
        constants  = ', '.join([f'constant("{str(arg)}")' for arg in positive_goal.args if str(arg)[0].isupper()])
        if len(goal.args) > 0:
            if not variables:
                goal_predicate = f'predicate(("{positive_goal.op}", {constants}))'
                goal_value = f'value(predicate(("{positive_goal.op}", {constants})), {"false" if positive_goal != goal else "true"})'
                asp_facts += f'goal({goal_predicate}, {goal_value}).\n'
            # TODO: handle existentially quantified case where goals contain variables
        else:
            goal_predicate = f'predicate(("{positive_goal.op}"))'
            goal_value = f'value({goal_predicate}, {"false" if positive_goal != goal else "true"})'
            asp_facts += f'goal({goal_predicate}, {goal_value}).\n'
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
    if matches[1:]:
        args = f'({",".join(matches[1:])})'
    else:
        args = ''
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
    for t in range(1, t_max + 1):

        time_bounded_program = f'#const horizon={t}.\n{asp_planning_program}'''

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
