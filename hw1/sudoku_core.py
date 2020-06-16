"""
    Algorithms to solve Sudoku on a (k*k)^2 size grid, for varying sizes of k

    Author: Michael Neely
    Student ID: 12547190

    Algorithms implemented:
        - Propagation
        - Satisfiability (SAT) solver encoding with PySAT
            https://github.com/pysathq/pysat
        - Constraint Satisfaction Problem (CSP) encoding with Google OR-Tools
            https://developers.google.com/optimization
        - Answer Set Programming (ASP) encoding with Potassco clingo
            https://potassco.org/clingo/
        - Integer Linear Programming (ILP) encoding with Gurobi
            https://www.gurobi.com/

    References:
        - Artificial Intelligence: A Modern Approach, 3rd Global Edition by Stuart Russell and Peter Norvig
        - Answer Set Solving in Practice by Martin Gebser, Roland Kaminski, Benjamin Kaufmann and Torsten Schaub
        - Optimized CNF encoding for sudoku puzzles (Kwon and Jain, 2006)
            https://www.researchgate.net/publication/228566840_Optimized_CNF_encoding_for_sudoku_puzzles
        - Answer set programming: Sudoku solver by Enrico Höschler
            https://ddmler.github.io/asp/2018/07/10/answer-set-programming-sudoku-solver.html
        - Official Gurobi 9.0 documentation Sudoku example
            https://www.gurobi.com/documentation/9.0/examples/sudoku_py.html
"""

# Standard library
import collections
from copy import copy
import itertools
from typing import Generator, List, Tuple, Union

# External imports
import clingo
import gurobipy as gp
from ortools.sat.python import cp_model
from pysat.formula import CNF, IDPool
from pysat.solvers import MinisatGH

###
### Sudoku helper functions
###
def get_cell_indices(k: int) -> Generator[Tuple[int, int], None, None]:
    """Generate all possible (row, column) cell indices for a k-by-k block Sudoku puzzle

    :param k: Size of the Sudoku puzzle (a grid of k x k blocks)
    :type k: int
    :return: Yields (row index, column index) tuples
    :rtype: Generator[Tuple[int, int], None, None]
    """
    grid_size = k**2
    return itertools.product(range(grid_size), range(grid_size))


def get_cell_indices_by_block(k: int) -> Generator[List[Tuple[int, int]], None, None]:
    """Generate lists of (row, column) cell indices for each block in a k-by-k block Sudoku puzzle

    :param k: Size of the Sudoku puzzle (a grid of k x k blocks)
    :type k: int
    :yield: Yields one list of (row, column) cell indices for each block
    :rtype: Generator[List[Tuple[int, int]], None, None]
    """
    for block_row in range(k):
        for block_column in range(k):
            block_indexes = itertools.product(
                range(k * block_row, k * block_row + k),
                range(k * block_column, k * block_column + k))
            yield block_indexes


def get_block_number(row_index: int, column_index: int, k: int) -> int:
    """Retrieve the block number corresponding to the given row and column indices for a k-by-k block Sudoku puzzle
       For example, block 0 in a k=3 Sudoku puzzle contains the indices (0,0), (0,1), (0,2), (1,0) ... (2,2)

    :param row_index: Row index
    :type row_index: int
    :param column_index: Column index
    :type column_index: int
    :param k: Size of the Sudoku puzzle (a grid of k x k blocks)
    :type k: int
    :return: Corresponding block number (between 0 and k**2-1)
    :rtype: int
    """
    block_row = row_index // k
    block_column = column_index // k
    return (block_row * k) + block_column


def solve_empty_puzzle_with_csp_alldiff(k: int) -> Union[List[List[int]], None]:
    """Solve a k-by-k block empty Sudoku puzzle by encoding the problem as a collection of AllDiff constraints
       and using a constraint satisfaction problem (CSP) solver per section 6.2.6 of Russell and Norvig

    :param k: Size of the Sudoku puzzle (a grid of k x k blocks)
    :type k: int
    :return: The solved sudoku puzzles as a list of lists or None if no solution is found.
    :rtype: Union[List[List[int]], None]
    """

    grid_size = k**2

    # initialize empty board
    solved_sudoku = [[0 for r in range(grid_size)] for c in range(grid_size)]

    model = cp_model.CpModel()

    cell_indices = list(get_cell_indices(k))

    # variable are identified by {row_index}-{column_index}, e.g. 8-4
    variables = {}
    for row_index, column_index in cell_indices:
        variables[f'{row_index}-{column_index}'] = model.NewIntVar(1, grid_size, f'{row_index}-{column_index}')

    for row_index in range(k**2):
        # Row uniqueness
        model.AddAllDifferent([variables[f'{row_index}-{c}'] for c in range(k**2)])
        # Column uniqueness
        model.AddAllDifferent([variables[f'{c}-{row_index}'] for c in range(k**2)])

    # Block uniqueness
    for block_indices in get_cell_indices_by_block(k):
        model.AddAllDifferent([variables[f'{r}-{c}'] for r, c in block_indices])

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if not status == cp_model.FEASIBLE:
        return None

    # Fill in solutions
    for row_index, column_index in cell_indices:
        solved_sudoku[row_index][column_index] = solver.Value(variables[f'{row_index}-{column_index}'])

    return solved_sudoku


def solve_partial_puzzle_with_csp_binary(sudoku: List[List[int]], k: int) -> Union[List[List[int]], None]:
    """Solve a k-by-k block partially filled Sudoku puzzle by encoding the problem as a collection of binary
       constraints and using a constraint satisfaction problem (CSP) solver similar to the SAT approach

    :param sudoku: A Sudoku puzzle represented as a list of lists. Values to be filled in are set to zero (0)
    :type sudoku: List[List[int]]
    :param k: Size of the Sudoku puzzle (a grid of k x k blocks)
    :type k: int
    :return: The solved sudoku puzzles as a list of lists or None if no solution is found.
    :rtype: Union[List[List[int]], None]
    """
    solved_sudoku = copy(sudoku)

    grid_size = k**2
    cell_indices = list(get_cell_indices(k))

    model = cp_model.CpModel()

    # Binary variable are identified by {row_index}-{column_index}-{value} all starting at 0, e.g. 8-4-0
    variables = collections.defaultdict(lambda: collections.defaultdict(dict))

    # Initialize binary variables
    for row_index, column_index in cell_indices:
        existing_val = solved_sudoku[row_index][column_index]
        for val in range(grid_size):
            if existing_val:
                if val + 1 == existing_val:
                    variables[row_index][column_index][val] = model.NewConstant(1)
                else:
                    variables[row_index][column_index][val] = model.NewConstant(0)
            else:
                variables[row_index][column_index][val] = model.NewBoolVar(f'{row_index}-{column_index}-{val}')

    # Binary encoding - row/column/cell level
    for row_index, cell_index in cell_indices:
        # Definedness
        model.AddBoolOr([variables[row_index][cell_index][v] for v in range(grid_size)])
        model.AddBoolOr([variables[row_index][v][cell_index] for v in range(grid_size)])
        model.AddBoolOr([variables[v][cell_index][row_index] for v in range(grid_size)])

        for val_index_i, val_index_j in itertools.combinations(range(grid_size), 2):
            # Uniqueness
            model.AddBoolOr([
                variables[row_index][cell_index][val_index_i].Not(),
                variables[row_index][cell_index][val_index_j].Not()])
            model.AddBoolOr([
                variables[row_index][val_index_i][cell_index].Not(),
                variables[row_index][val_index_j][cell_index].Not()])
            model.AddBoolOr([
                variables[val_index_i][row_index][cell_index].Not(),
                variables[val_index_j][row_index][cell_index].Not()])

    # Binary encoding - block level
    for val in range(grid_size):
        for block_indices in get_cell_indices_by_block(k):
            # Block definedness
            model.AddBoolOr([variables[r][c][val]  for r, c in block_indices])

            # Block uniqueness
            for block_index_tuple in itertools.combinations(block_indices, 2):
                (row_1, column_1), (row_2, column_2) = block_index_tuple
                model.AddBoolOr([
                    variables[row_1][column_1][val].Not(),
                    variables[row_2][column_2][val].Not()])

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if not status == cp_model.FEASIBLE:
        return None

    # Fill in solution
    for row_index, column_index in cell_indices:
        for val in range(grid_size):
            if solver.Value(variables[row_index][column_index][val]):
                solved_sudoku[row_index][column_index] = val + 1

    return solved_sudoku


###
### Propagation function to be used in the recursive sudoku solver
###
def propagate(sudoku_possible_values: List[List[List[int]]], k: int) -> List[List[List[int]]]:
    """Filter out impossible values from the domain of each Sudoku cell using the basic rules of
       the game: no duplicates in any unit (row, column, or block).

    :param sudoku_possible_values: Current possible values for every cell in a sudoku puzzle
    :type sudoku_possible_values: List[List[List[int]]]
    :param k: Size of the Sudoku puzzle (a grid of k x k blocks)
    :type k: int
    :return: A (hopefully) filtered list of possible values for every cell
    :rtype: List[List[List[int]]]
    """

    filtered_possible_values = copy(sudoku_possible_values)

    grid_size = k**2

    cell_indices = list(get_cell_indices(k))

    # Filtering is not possible if the puzzle has no pre-filled cells
    is_empty = all([len(filtered_possible_values[r][c]) == grid_size for r, c in cell_indices])
    if is_empty:
        return filtered_possible_values

    # Record impossible values at unit (row, column, block) indices
    disallowed_at_row = collections.defaultdict(set)
    disallowed_at_column = collections.defaultdict(set)
    disallowed_at_block = collections.defaultdict(set)

    for row_index, column_index in cell_indices:
        possible_values = filtered_possible_values[row_index][column_index]
        if len(possible_values) == 1:
            cell_value = possible_values[0]
            block_number = get_block_number(row_index, column_index, k)

            disallowed_at_row[row_index].add(cell_value)
            disallowed_at_column[column_index].add(cell_value)
            disallowed_at_block[block_number].add(cell_value)

    # Reduce to plausible domains
    for row_index, column_index in cell_indices:
        possible_values = filtered_possible_values[row_index][column_index]
        if len(possible_values) != 1:
            block_number = get_block_number(row_index, column_index, k)
            logical_values = set(range(1, grid_size +1))
            logical_values -= disallowed_at_column[column_index]
            logical_values -= disallowed_at_row[row_index]
            logical_values -= disallowed_at_block[block_number]
            filtered_possible_values[row_index][column_index] = list(logical_values)

    # More could be done: i.e., naked triples and other sudoku domain reduction strategies

    return filtered_possible_values


###
### Solver that uses SAT encoding
###
def solve_sudoku_SAT(sudoku: List[List[int]], k: int) -> Union[List[List[int]], None]:
    """Solve a k-by-k block Sudoku puzzle by encoding the problem as a propositional CNF formula
       and using a satisfiability (SAT) solver

    :param sudoku: A Sudoku puzzle represented as a list of lists. Values to be filled in are set to zero (0)
    :type sudoku: List[List[int]]
    :param k: Size of the Sudoku puzzle (a grid of k x k blocks)
    :type k: int
    :return: The solved sudoku puzzles as a list of lists or None if no solution is found.
    :rtype: Union[List[List[int]], None]
    """
    # Record a (row, column, value) triple as a Variable ID in a PySAT ID Pool
    def triplet_to_id(row_index: int, column_index: int, val: int) -> int:
        # indexes need to start at 1 in the id pool
        return id_pool.id(tuple([row_index + 1, column_index + 1, val]))

    # Extract the (row, column, value) triple from the PySAT Variable ID
    def id_to_triplet(vid: int) -> Tuple[int, int, int]:
        row, column, val = id_pool.obj(vid)
        # convert back to indexes
        return row - 1, column - 1, val

    solved_sudoku = copy(sudoku)

    grid_size = k**2
    cell_indices = list(get_cell_indices(k))

    formula = CNF()
    id_pool = IDPool()

    # Build the CNF Formula using the definition of Kwon and Jain (2006)
    # (https://www.researchgate.net/publication/228566840_Optimized_CNF_encoding_for_sudoku_puzzles)
    # Extended Encoding = cell definedness AND cell uniqueness AND row definedness AND row uniqueness
    #   AND column definedness AND column uniqueness AND block definedness
    #   AND block uniqueness AND assigned cells
    # definedness: each (cell, row, column, block) has at least one number from 1 to grid_size+1
    # uniqueness each (cell, row, column, block) has at most one number from 1 to grid_size+1

    # Row/column/cell level
    # You can swap indices to declare uniqueness and definedness in a single loop
    for row_index, column_index in cell_indices:
        # Cell definedness
        formula.append([triplet_to_id(row_index, column_index, v) for v in range(grid_size)])

        # Row definedness
        formula.append([triplet_to_id(row_index, v, column_index) for v in range(grid_size)])

        # Column definedness
        formula.append([triplet_to_id(v, column_index, row_index) for v in range(grid_size)])

        # Assigned cells
        val = sudoku[row_index][column_index]
        if val != 0:
            formula.append([triplet_to_id(row_index, column_index, val - 1)])

        for val_index_i, val_index_j in itertools.combinations(range(grid_size), 2):
            # Cell uniqueness
            formula.append([
                -triplet_to_id(row_index, column_index, val_index_i),
                -triplet_to_id(row_index, column_index, val_index_j)])

            # Row uniqueness
            formula.append([
                -triplet_to_id(row_index, val_index_i, column_index),
                -triplet_to_id(row_index, val_index_j, column_index)])

            # Column uniqueness
            formula.append([
                -triplet_to_id(val_index_i, row_index, column_index),
                -triplet_to_id(val_index_j, row_index, column_index)])

    # Block level
    for val in range(grid_size):
        for block_indices in get_cell_indices_by_block(k):
            # Block definedness
            formula.append([triplet_to_id(r, c, val) for r, c in block_indices])

            # Block uniqueness
            for block_index_tuple in itertools.combinations(block_indices, 2):
                (row_1, column_1), (row_2, column_2) = block_index_tuple
                formula.append([
                    -triplet_to_id(row_1, column_1, val),
                    -triplet_to_id(row_2, column_2, val)])

    # Solve
    solver = MinisatGH()
    solver.append_formula(formula)
    answer = solver.solve()

    if not answer:
        return None

    # Fill in solution
    model = solver.get_model()
    for vid in model:
        if vid > 0:
            row_index, column_index, val = id_to_triplet(vid)
            solved_sudoku[row_index][column_index] = val + 1

    return solved_sudoku


###
### Solver that uses CSP encoding
###
def solve_sudoku_CSP(sudoku: List[List[int]], k: int) -> Union[List[List[int]], None]:
    """Solve a k-by-k block Sudoku puzzle by encoding the problem as a collection of constraints
       and using a constraint satisfaction problem (CSP) solver

    :param sudoku: A Sudoku puzzle represented as a list of lists. Values to be filled in are set to zero (0)
    :type sudoku: List[List[int]]
    :param k: Size of the Sudoku puzzle (a grid of k x k blocks)
    :type k: int
    :return: The solved sudoku puzzles as a list of lists or None if no solution is found.
    :rtype: Union[List[List[int]], None]
    """

    # If the puzzle is empty, the AllDiff approach is faster
    is_empty = sum(sum(r) for r in sudoku) == 0
    if is_empty:
        solved_sudoku = solve_empty_puzzle_with_csp_alldiff(k)
    else:
        solved_sudoku = solve_partial_puzzle_with_csp_binary(sudoku, k)

    return solved_sudoku


###
### Solver that uses ASP encoding
###
def solve_sudoku_ASP(sudoku: List[List[int]], k: int) -> Union[List[List[int]], None]:
    """Solve a k-by-k block Sudoku puzzle by encoding it as an answer set programming (ASP) problem
       and using an ASP solver

    :param sudoku: A Sudoku puzzle represented as a list of lists. Values to be filled in are set to zero (0)
    :type sudoku: List[List[int]]
    :param k: Size of the Sudoku puzzle (a grid of k x k blocks)
    :type k: int
    :return: The solved sudoku puzzles as a list of lists or None if no solution is found.
    :rtype: Union[List[List[int]], None]
    """
    ###
    ### Solution Notes
    ###
    ### Inspiration from the following references:
    ###     - Chapter 3 of Answer Set Solving in Practice
    ###     - 'block' helper function retrieved from the blog of Enrico Höschler:
    ###         https://ddmler.github.io/asp/2018/07/10/answer-set-programming-sudoku-solver.html
    ###
    ### sudoku(R, C, V) denotes a sudoku cell at a row index (R) and column index (C) filled with value (V)
    ###
    ### There are n == k**2 such rows, columns, and values
    ###
    ### Rows and columns are indexed at zero (0). Values start at one (1)
    ###
    ### Constraints:
    ###     - Cardinality and Choice: Each cell must have exactly one value in the domain [1,n]
    ###     - Integrity: No cell in the same row may share a value
    ###     - Integrity: No cell in the same column may share a value
    ###     - Integrity: No cell in the same block may share a value

    solved_sudoku = copy(sudoku)

    grid_size = k**2

    # Build a string where each line is the fact corresponding to a pre-filled cell
    facts = ''
    for row_index, column_index in get_cell_indices(k):
        value = solved_sudoku[row_index][column_index]
        if value != 0:
            facts += f'sudoku({row_index}, {column_index}, {value}).\n'

    # As noted above, this was inspired by Enrico Höschler's blog
    # The rule applies if two cells are in the same block
    same_block_rule = 'block(R1, C1, R2, C2) :- row(R1), row(R2), column(C1), column(C2), R1/k == R2/k, C1/k == C2/k.'

    # Constraints
    cell_uniqueness = '1{sudoku(R, C, V): value(V)}1 :- row(R) , column(C).'
    row_integrity = ':- sudoku(R, C1, V), sudoku(R, C2, V), C1 != C2.'
    column_uniqueness = ':- sudoku(R1, C, V), sudoku(R2, C, V), R1 != R2.'
    block_uniqueness = ':- sudoku(R1, C1, V), sudoku(R2, C2, V), block(R1, C1, R2, C2), R1 != R2, C1 != C2.'

    asp_program = f"""
        #const k = {k}.
        #const n = {grid_size}.

        row(0..n-1).
        column(0..n-1).
        value(1..n).

        {facts}

        {same_block_rule}

        {cell_uniqueness}
        {row_integrity}
        {column_uniqueness}
        {block_uniqueness}
    """

    # Fill in the puzzle once an answer set is found
    def on_model(model):
        for atom in model.symbols(atoms=True):
            if atom.name == 'sudoku':
                row_index, column_index, value = atom.arguments
                solved_sudoku[row_index.number][column_index.number] = value.number

    # Encode and solve
    control = clingo.Control()
    control.add("base", [], asp_program)
    control.ground([("base", [])])
    control.configuration.solve.models = 1 # we only need one feasible solution
    answer = control.solve(on_model=on_model)

    if not answer.satisfiable:
        return None

    return solved_sudoku


###
### Solver that uses ILP encoding
###
def solve_sudoku_ILP(sudoku: List[List[int]], k: int) -> Union[List[List[int]], None]:
    """Solve a k-by-k block Sudoku puzzle by encoding it as an integer linear programming (ILP) problem
       and using an ILP optimizer

    :param sudoku: A Sudoku puzzle represented as a list of lists. Values to be filled in are set to zero (0)
    :type sudoku: List[List[int]]
    :param k: Size of the Sudoku puzzle (a grid of k x k blocks)
    :type k: int
    :return: The solved sudoku puzzles as a list of lists or None if no solution is found.
    :rtype: Union[List[List[int]], None]
    """
    ###
    ### Solution Notes
    ###
    ### Inspiration from the official Gurobi documentation Sudoku example:
    ###     https://www.gurobi.com/documentation/9.0/examples/sudoku_py.html
    ###

    solved_sudoku = copy(sudoku)

    grid_size = k**2

    cell_indices = list(get_cell_indices(k))

    model = gp.Model()

    # Initialize variables
    variables = model.addVars(grid_size, grid_size, grid_size, vtype=gp.GRB.BINARY, name='S')
    for row_index, column_index in cell_indices:
        value = solved_sudoku[row_index][column_index]
        if value != 0:
            variables[row_index, column_index, value - 1].LB = 1

    # Cell uniqueness
    model.addConstrs((variables.sum(r, c, '*') == 1 for r, c in cell_indices), name='V')
    # Row uniqueness
    model.addConstrs((variables.sum(r, '*', c) == 1 for r, c in cell_indices), name='R')
    # Column uniqueness
    model.addConstrs((variables.sum('*', r, c) == 1 for r, c in cell_indices), name='C')

    # Block uniqueness
    model.addConstrs((
        gp.quicksum(
            variables[r, c, v] for r, c in block_indices) == 1
        for v in range(grid_size)
        for block_indices in get_cell_indices_by_block(k)))

    model.optimize()

    if model.status != gp.GRB.OPTIMAL:
        return None

    # Fill in the puzzle
    solution = model.getAttr('X', variables)
    for row_index, column_index in cell_indices:
        for value in range(grid_size):
            if solution[row_index, column_index, value] > 0.5:
                solved_sudoku[row_index][column_index] = value + 1

    return solved_sudoku
