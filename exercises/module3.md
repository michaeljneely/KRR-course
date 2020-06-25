# Exercises related to Automated Planning

These are exercise questions about *automated planning* that you can use
to get a better understanding of the material related to this topic.
The solutions to these exercise questions are not published here, but you are free
to ask about the solutions (e.g., on the [course Canvas page](https://canvas.uva.nl/courses/10768))
if you get stuck and you are free to discuss these exercise questions and their solutions
with other students in the course.

Moreover, these exercise questions are indicative of the type of questions that
you can expect on the final exam.

---

## Exercise 3.1: Encoding 3SAT into planning

### Exercise 3.1.a

The class 3CNF consists of all propositional logic formulas &varphi;
in conjunctive normal form (CNF) where each clause of &varphi;
contains at most three literals.
The problem 3SAT is the problem of deciding whether a given propositional
logic formula &varphi; in 3CNF is satisfiable.

Show how the problem of 3SAT can efficiently be encoded into the
problem of finding whether a given planning problem has a plan.
That is, describe an algorithm, that takes less than exponential time,
that takes as input a 3CNF formula &varphi;,
and that produces a planning problem for which there exists a
sequence of actions that achieves the goal when applied to the initial state
if and only if &varphi; is satisfiable.

*Hint:*
- For a 3CNF formula with propositional variables *x<sub>1</sub>,...,x<sub>n</sub>*
and clauses *c<sub>1</sub>,...,c<sub>m</sub>*, create a planning problem that uses
propositions `X(1),...,X(n),Y(1),...,Y(n),C(1),...,C(m)`.

For each i, have two actions:

- MakeTrue(i), with precondition: ~X(i) & ~Y(i), effect: X(i)
- MakeFalse(i), with precondition: ~X(i) & ~Y(i), effect: Y(i)

For each literal l in clause j:

- if l is a positive literal x_i, then have an action SatisfyClause(l,j), with precondition X(i), and effect: C(j)
- if l is a negative literal ~x_i, then have an action SatisfyClause(l,j), with precondition Y(i), and effect C(j)

Initial state: \emptyset

Goal: C(1) & ... & C(m)

Then you can argue that there is a sequence of actions that leads to the goal if and only if there is a truth assignment that makes the formula with clauses c_1, …, c_m true.

---

## Exercise 3.2: Modelling

### Exercise 3.2.a

Consider the following planning scenario.
There are two drones on a 4x4 grid.
Each location of the grid either contains (i) a warehouse, (ii) a delivery location, or (iii) nothing.
Moreover, above each location on the grid can be at most one drone at the same time.
The drones can move to adjacent locations on the grid.
If the drone is above a warehouse, it can pick up a package.
If the drone is above a delivery location, it can deliver a package (if it is carrying one).
Each drone can carry at most one package at a time.
All packages are identical.
The warehouse has an unlimited supply of packages.
The (initial) setup is as follows (where every cell contains nothing, unless specified otherwise):
- Cell (0,0): a warehouse and above it a drone
- Cell (0,1): above it a drone
- Cell (1,2): a delivery location
- Cell (2,1): a delivery location
- Cell (3,3): a delivery location

The goal is to have delivered exactly package at each delivery location.

Show how to model this scenario in the PDDL planning language
(as used in [[Russell, Norvig, 2016]](https://github.com/rdehaan/KRR-course#aima)).

```
# Initial state
initial: (At(W1, 0, 0) & At(D1, 0, 0) & At(D2, 0, 1) & At(L1, 1, 2) & At(L2, 2, 1) & At(L3, 3, 3) & Cell(0, 0) & Cell(0, 1) & Cell(0, 2) & Cell(0, 3) & Cell(1, 0) & Cell(1, 1) & Cell(1, 2) & Cell(1, 3) & Cell(2, 0) & Cell(2, 1) & Cell(2, 2) & Cell(2, 3) & Cell(3, 0) & Cell(3, 1) & Cell(3, 2) & Cell(3, 3) & Drone(D1) & Drone(D2) & Location(L1) & Location(L2) & Location(L3))
# Goals
goals: (Delivered(L1) & Delivered(L2) & Delivered(L3))
# Action Load(d, r, c)
action: Load(d, r, c); (At(d, r, c) & At(W1, r, c) & Drone(d) & ~Full(d)); (Full(d))
# Action Unload(d, l, r, c)
action: Unload(d, l, r, c); (At(d, r, c) & At(l, r, c) & Drone(d) & Location(l) & Full(d) & ~Delivered(l)); (Delivered(l) & ~Full(d))
# Action FlyDown(d1, d2, r, c)
action: FlyDown(d1, d2, r, c); (At(d1, r, c) & ~At(d2, r + 1, c) & Cell(r + 1, c) & Drone(d1) & Drone(d2)); (At(d1, r + 1, c) & ~At(d1, r, c))
... Same idea for FlyUp, FlyLeft, FlyRight
```
---

## Exercise 3.3: Planning with two (or more) goals

### Exercise 3.3.a

Consider the following general variant of the classical planning setup.
You have an initial state *I*, and a set *A* of actions (each deterministic, with a precondition and an effect),
specified in the PDDL planning language (as used in [[Russell, Norvig, 2016]](https://github.com/rdehaan/KRR-course#aima)).
In addition to this, you get two goals *G<sub>1</sub>* and *G<sub>2</sub>*, both consisting of a set of statements.
In other words, the only difference with the typical classical planning setup is that instead of a single *G* specifying
the goals, you have two: *G<sub>1</sub>* and *G<sub>2</sub>*.
The task in this setting is to find a sequence of actions such that for both *G<sub>1</sub>* and *G<sub>2</sub>*
it holds that they achieved at some point. In other words, if you apply the sequence of actions to the initial state,
you get a resulting sequence of states, and in (at least) one of these states *G<sub>1</sub>* is achieved,
and in (at least) one of these states *G<sub>2</sub>* is achieved.

Show how you can model this scenario with two goal specifications, *G<sub>1</sub>* and *G<sub>2</sub>*,
in the classical setting with only one goal specification *G*.
In other words, describe how to make changes to *I*, *A*, *G<sub>1</sub>* and *G<sub>2</sub>*,
so that you get a classical planning problem, for which there exists a plan (that achieves *G*)
if and only if for the 'double-goal' planning problem there exists a sequence of actions that achieves
*G<sub>1</sub>* and *G<sub>2</sub>* (not necessarily at the same time, and in any order).

*Hints:*
- Use additional statements such as `Achieved(Goal1)` and `Achieved(Goal2)` that are false in the initial state.
- Add actions that can be used to make these statements true (under the right conditions).
- Specify the 'unified' goal *G* using these statements `Achieved(Goal1)` and `Achieved(Goal2)`.

Answer: add an AchieveGoalX action per goal, with preconditions equivalent to the preconditions of that goal and an effect Achieved(GoalX).

### Exercise 3.3.b

Consider the same question as in Exercise 3.3.a, with the difference that *G<sub>1</sub>*
must be achieved not later than *G<sub>2</sub>*.
In other words, the modified planning setup concerns the task of finding a sequence of
actions such that:
- if you apply the sequence of actions to the initial state,
you get a resulting sequence of states *s<sub>0</sub>,...,s<sub>m</sub>*,
- there is a state *s<sub>i</sub>* in this sequence *s<sub>0</sub>,...,s<sub>m</sub>*
in which *G<sub>1</sub>* is achieved, and
- there is a state *s<sub>j</sub>* *s<sub>0</sub>,...,s<sub>m</sub>*
with *i &le; j* in which *G<sub>2</sub>* is achieved.

Show how you can model this scenario with two goal specifications,
*G<sub>1</sub>* and *G<sub>2</sub>*,
in the classical setting with only one goal specification *G*.

Answer: Extend 3.3.a by simply adding Achieved(Goal1) as a precondition to the AchieveGoal2 action.
### Exercise 3.3.c

Consider the questions of Exercises 3.3.a and 3.3.b,
but then generalized to more than two goals: *G<sub>1</sub>, ..., G<sub>k</sub>*.
Show how to answer the questions of Exercises 3.3.a and 3.3.b
for an arbitrary number *k* of goal specifications.

Answer: Extend as per 3.3.a and 3.3.b, where, for every goal_i, add a precondition specifying that every goal from goal_1..goal_i-1 should be achieved.
