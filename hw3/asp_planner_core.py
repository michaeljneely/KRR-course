"""
    An algorithm to solve sequential planning problems (specified in a format unique to this assignment) with Clingo

    Author: Michael Neely
    Student ID: 12547190

    References:
        - plasp 3: Towards Effective ASP Planning (Dimopoulos et al. 2018)
            https://arxiv.org/pdf/1812.04491.pdf
        - Potassco plasp
            https://github.com/potassco/plasp/
"""

# Standard library
import collections
import re
from typing import List, Union

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

       All credit goes to the plasp authors. My only contribution is explaining each group of statements in detail

    :return: Initialized ASP sequential planning program, ready to be extended with facts from a planning problem
    :rtype: str
    """
    # We reason about the state of the world at particular time steps: [0, t_max]
    asp_planning_program = 'time(0..horizon).\n'

    # Variables contain values, which may be True or False
    asp_planning_program += 'boolean(true).\n'
    asp_planning_program += 'boolean(false).\n'
    # The contains/2 atom captures this relationship
    asp_planning_program += 'contains(X, value(X, B)) :- variable(X), boolean(B).\n'

    # The initial state is at time t=0
    # The holds/3 atom captures the value of a variable at a particular timestep t >= 0
    asp_planning_program += 'holds(Variable, Value, 0) :- initialState(Variable, Value).\n'

    # Closed World Assumption (CWA): Any ground atoms in the initial state which are not explicitly declared True
    # are set to False
    asp_planning_program += 'initialState(X, value(X, false)) :- variable(X), not initialState(X, value(X, true)).\n'

    # The solution to the planning problem is extracted from occurs/2 atoms
    # This is a sequential encoding: only one action may occur at a particular timestep
    # Also, actions may only occur AFTER the initial state.
    asp_planning_program += '1 {occurs(Action, T) : action(Action)} 1 :- time(T), T > 0.\n'

    # An action may not occur unless its preconditions are met (i.e., for an action to occur at time t,
    # all applicable variables must hold the values specified in the precondition at time t-1)
    asp_planning_program += '''
        :- occurs(Action, T), precondition(Action, Variable, Value), not holds(Variable, Value, T - 1).\n'''

    # Capture the effects of an action: at time t, the value of a variable is changed to the one specified in the
    # action's postcondition as long as the action was valid (see previous statement).
    asp_planning_program += '''
        caused(Variable, Value, T) :-
            occurs(Action, T),
            postcondition(Action, Effect, Variable, Value),
            holds(VariablePre, ValuePre, T - 1) : precondition(Effect, VariablePre, ValuePre).\n
    '''

    # A variable is considered modified if its value was changed by an action
    asp_planning_program += 'modified(Variable, T) :- caused(Variable, Value, T).\n'

    # The so-called 'inertia' statements. At a particular timestep, the value of a variable was either:
    # 1) Modified and therefore holds a new value
    asp_planning_program += 'holds(Variable, Value, T) :- caused(Variable, Value, T).\n'
    # 2) Was not modified and therefore continues to hold its previous value
    asp_planning_program += 'holds(variable(V), Value, T) :- holds(variable(V), Value, T - 1), not modified(variable(V), T), time(T).\n'

    # The goal is not met unless the appropriate variables hold their goal values at the final timestep
    asp_planning_program += ':- goal(Variable, Value), not holds(Variable, Value, horizon).\n'

    return asp_planning_program


def planning_problem_to_asp_facts(planning_problem: PlanningProblem) -> str:
    """Translate a pseudo-PDDL description to a collection of ASP facts per an approach similar to 
       plasp translate (https://github.com/potassco/plasp)

    :param planning_problem: A description of the initial state, action(s), and goal(s) of a planning problem
    :type planning_problem: PlanningProblem
    :return: A single string containing the ASP fact translations
    :rtype: str
    """
    variables_string = ''
    constants_string = ''
    initial_state_string = ''
    actions_string = ''
    goal_string = ''

    ###
    seen_predicates = set()
    seen_constants = set()

    for predicate in planning_problem.initial:
        if predicate.op not in seen_predicates:
            num_arguments = len(predicate.args)
            predicate_variables = ', '.join([f'X{i}' for i in range(1, num_arguments +1)])
            predicate_types = ', '.join([f'has(X{i}, type("object"))' for i in range(1, num_arguments +1)])
            variables_string += f'variable(variable("{predicate.op}", {predicate_variables})) :- {predicate_types}.\n'
            seen_predicates.add(predicate.op)

        constants  = ', '.join([f'constant("{str(arg)}")' for arg in predicate.args])
        initial_state_variable = f'variable(("{predicate.op}", {constants}))'
        initial_state_value = f'value(variable(("{predicate.op}", {constants})), true)'
        initial_state_string += f'initialState({initial_state_variable}, {initial_state_value}).\n'

        for arg in predicate.args:
            if arg not in seen_constants:
                seen_constants.add(arg)
    
    for goal in planning_problem.goals:

        neg = False

        if goal.op == '~':
            neg = True
            goal = Expr(goal.args[0].op, *goal.args[0].args)

        for arg in goal.args:
            if arg not in seen_constants and str(arg)[0].isupper():
                seen_constants.add(arg)
        constants  = ', '.join([f'constant("{str(arg)}")' for arg in goal.args if str(arg)[0].isupper()])
        if constants:
            goal_variable = f'variable(("{goal.op}", {constants}))'
            goal_value = f'value(variable(("{goal.op}", {constants})), {"false" if neg else "true"})'
            goal_string += f'goal({goal_variable}, {goal_value}).\n'

    for constant in seen_constants:
        constants_string += f'constant(constant("{constant}")).\n'
        constants_string += f'has(constant("{constant}"), type("object")).\n'

    for action in planning_problem.actions:
        arg_map = {}
        num_arguments = len(action.args)
        for i, arg in enumerate(action.args):
            arg_map[str(arg)] = f'X{i+1}'

        action_variables = ', '.join(arg_map.values())
        action_types = ', '.join([f'has({k}, type("object"))' for k in arg_map.values()])
        actions_string += f'action(action(("{action.name}", {action_variables}))) :- {action_types}.\n'
        
        for precondition in action.precond:

            neg = False

            if precondition.op == '~':
                neg = True
                precondition = Expr(precondition.args[0].op, *precondition.args[0].args)

            precondition_action = f'action(("{action.name}", {", ".join(arg_map.values())}))'

            action_arg_map = {}
            for arg in precondition.args:
                if str(arg) in arg_map:
                    action_arg_map[str(arg)] = arg_map[str(arg)]
                else:
                    action_arg_map[str(arg)] = f'constant("{str(arg)}")'
            # precondition(action(("load", X1, X2, X3)), variable(("at-airport", X1, X3)), value(variable(("at-airport", X1, X3)), true)) :- action(action(("load", X1, X2, X3))).
           
            precondition_variable = f'variable(("{precondition.op}", {", ".join(action_arg_map[str(arg)] for arg in precondition.args)}))'
            precondition_value = f'value(variable(("{precondition.op}", {", ".join(action_arg_map[str(arg)] for arg in precondition.args)})), {"false" if neg else "true"})'

            actions_string += f'precondition({precondition_action}, {precondition_variable}, {precondition_value}) :- action({precondition_action}).\n'

        for postcondition in action.effect:

            neg = False

            if postcondition.op == '~':
                neg = True
                postcondition = Expr(postcondition.args[0].op, *postcondition.args[0].args)

            action_arg_map = {}
            for arg in postcondition.args:
                if str(arg) in arg_map:
                    action_arg_map[str(arg)] = arg_map[str(arg)]
                else:
                    action_arg_map[str(arg)] = f'constant("{str(arg)}")'

            postcondition_action = f'action(("{action.name}", {", ".join(arg_map.values())}))'
            postcondition_variable = f'variable(("{postcondition.op}", {", ".join(action_arg_map[str(arg)] for arg in postcondition.args)}))'
            postcondition_value = f'value(variable(("{postcondition.op}", {", ".join(action_arg_map[str(arg)] for arg in postcondition.args)})), {"false" if neg else "true"})'
            actions_string += f'postcondition({postcondition_action}, effect(unconditional), {postcondition_variable}, {postcondition_value}) :- action({postcondition_action}).\n'

    facts = f'''

        % types
        type(type("object")).

        % variables
        {variables_string}

        % actions
        {actions_string}

        % objects
        {constants_string}

        % initial state
        {initial_state_string}

        % goal
        {goal_string}

        #show occurs/2.'''
    return facts


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
    args = ",".join(matches[1:])
    return f'{predicate}({args})'


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
    # Solution Notes
    #
    # "A scholar knows not to waste time rediscovering information already known" - Brandon Sanderson, The Way of Kings
    #
    # The goal is to convert a pseudo-PDDL description to ASP facts and then solve the planning problem.
    # A solution is found with a meta-encoding. "plasp 3: Towards Effective ASP Planning" (Dimopoulos et al. 2018 -
    # https://arxiv.org/pdf/1812.04491.pdf) details sequential and parallel meta-encodings, the source code for which
    # is available in the Plasp repository (https://github.com/potassco/plasp). The paper specifically refers to the
    # strips-incremental.lp encoding, but the authors also include a simplified sequential meta-encoding in
    # sequential-horizon.lp. This simplified encoding yields a plan as a sequence of time-stamped actions if and only
    # if the plan's length matches the externally defined horizon constant.
    #
    # At a high-level, my solution:
    #   1. Adds (and explains) a minimal version of plasp's sequential-horizon meta-encoding
    #   2. Converts the pseudo-PDDL description to ASP facts in a manner similar to the 'translate' feature of plasp
    #   3. Finds the optimal (shortest) plan by progressively incrementing the horizon from t = [1, t_max] (inclusive)
    #   4. Returns the optimal plan or None if no solution is found in the range [1, t_max]

    # Part 1: Add sequential horizon meta-encoding
    sequential_encoding = init_sequential_planning_program()

    # Part 2: Add ASP facts from the given planning_program
    facts = planning_problem_to_asp_facts(planning_problem)

    asp_planning_program = sequential_encoding + facts

    # If an answer set is found, build the solution by extracting the occurs atoms, sorting them by timestep, and
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
