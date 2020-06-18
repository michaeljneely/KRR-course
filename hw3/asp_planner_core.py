from planning import PlanningProblem, Action, Expr, expr
import planning
import re
import clingo
import collections

def extract_step(atom):
    text = str(atom)
    matches=re.findall(r'\"(.+?)\"',text)
    args = ",".join(matches[1:])
    return f'{matches[0]}({args})'

def solve_planning_problem_using_ASP(planning_problem,t_max):
    """
    If there is a plan of length at most t_max that achieves the goals of a given planning problem,
    starting from the initial state in the planning problem, returns such a plan of minimal length.
    If no such plan exists of length at most t_max, returns None.

    Finding a shortest plan is done by encoding the problem into ASP, calling clingo to find an
    optimized answer set of the constructed logic program, and extracting a shortest plan from this
    optimized answer set.

    Parameters:
        planning_problem (PlanningProblem): Planning problem for which a shortest plan is to be found.
        t_max (int): The upper bound on the length of plans to consider.

    Returns:
        (list(Expr)): A list of expressions (each of which specifies a ground action) that composes
        a shortest plan for planning_problem (if some plan of length at most t_max exists),
        and None otherwise.
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
    #   1. Converts the pseudo-PDDL description to ASP facts in a manner similar to the 'translate' feature of plasp
    #   2. Adds a minimal version of the sequential-horizon meta-encoding
    #   3. Finds the optimal (shortest) plan by progressively incrementing the horizon from t=[1, t_max] (inclusive)
    #   4. Returns the optimal plan or None if no solution is found in the range [1, t_max]

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
                
    asp_program = f'''
        % utils
        boolean(true).
        boolean(false).

        % types
        type(type("object")).

        % variables
        {variables_string}
        contains(X, value(X, B)) :- variable(X), boolean(B).

        % actions
        {actions_string}

        % objects
        {constants_string}

        % initial state
        {initial_state_string}
        initialState(X, value(X, false)) :- variable(X), not initialState(X, value(X, true)).

        % goal
        {goal_string}

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Horizon, must be defined externally
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        time(0..horizon).

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Establish initial state
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        holds(Variable, Value, 0) :- initialState(Variable, Value).

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Perform actions
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        1 {{occurs(Action, T) : action(Action)}} 1 :- time(T), T > 0.

        % Check preconditions
        :- occurs(Action, T), precondition(Action, Variable, Value), not holds(Variable, Value, T - 1).

        % Apply effects
        caused(Variable, Value, T) :-
            occurs(Action, T),
            postcondition(Action, Effect, Variable, Value),
            holds(VariablePre, ValuePre, T - 1) : precondition(Effect, VariablePre, ValuePre).

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Inertia rules
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        modified(Variable, T) :- caused(Variable, Value, T).

        holds(Variable, Value, T) :- caused(Variable, Value, T).
        holds(variable(V), Value, T) :- holds(variable(V), Value, T - 1), not modified(variable(V), T), time(T).

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Verify that goal is met
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        :- goal(Variable, Value), not holds(Variable, Value, horizon).

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        #show occurs/2.
    '''

    solution = []

    def on_model(model):
        nonlocal solution
        steps = [atom for atom in model.symbols(atoms=True) if atom.name == 'occurs']
        sorted_steps = sorted(steps, key=lambda tup: tup.arguments[1])
        solution = list(map(extract_step, sorted_steps))

    for t in range(1, t_max + 1):
        t_program = f'''

        #const horizon={t}.

        {asp_program}
        '''
        # Encode and solve
        control = clingo.Control()
        control.add("base", [], t_program)
        control.ground([("base", [])])
        control.configuration.solve.models = 1 # we only need one feasible solution
        answer = control.solve(on_model=on_model)

        if answer.satisfiable:
            return solution

    return None

