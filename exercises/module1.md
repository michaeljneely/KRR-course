# Exercises related to Problem Solving and Search

These are exercise questions about *problem solving and search* that you can use
to get a better understanding of the material related to this topic.
The solutions to these exercise questions are not published here, but you are free
to ask about the solutions (e.g., on the [course Canvas page](https://canvas.uva.nl/courses/10768))
if you get stuck and you are free to discuss these exercise questions and their solutions
with other students in the course.

Moreover, these exercise questions are indicative of the type of questions that
you can expect on the final exam.

---

## Exercise 1.1: DPLL

### Exercise 1.1.a

Show that the DPLL algorithm decides satisfiability of the
following propositional logic formula &varphi; in CNF
with only using unit propagation and pure literal elimination
(and thus without branching).

&varphi; consists of the following clauses:
```
(~x4 OR ~x5 OR ~x6)
(~x1 OR x2 OR x4)
(x4)
(~x4 OR x5)
(x7 OR x10)
(x1 OR x2)
(~x5 OR ~x7 OR ~x8)
(x2 OR ~x3)
(~x10)
(~x4 OR ~x5 OR x6 OR ~x7 OR x8 OR x9)
```

**Answer**:

With the DPLL algorithm, the given logic formula can be proven satisfiable through repeated application of pure literal elimination and unit propagation.

Iter 1: `x2` is a pure literal. Set `x2=True` and remove all clauses containing `x2`

```
(~x4 OR ~x5 OR ~x6)
(x4)
(~x4 OR x5)
(x7 OR x10)
(~x5 OR ~x7 OR ~x8)
(~x10)
(~x4 OR ~x5 OR x6 OR ~x7 OR x8 OR x9)
```

Iter 1: `x4` is a unit clause. Set `x4=True` and propagate:

```
(~x5 OR ~x6)
(x5)
(x7 OR x10)
(~x5 OR ~x7 OR ~x8)
(~x10)
(~x5 OR x6 OR ~x7 OR x8 OR x9)
```

Iter 1: `x9` is a pure literal. Set `x9=True` and remove all clauses containing `x9`

```
(~x5 OR ~x6)
(x5)
(x7 OR x10)
(~x5 OR ~x7 OR ~x8)
(~x10)
```

Iter 2: `x5` is a unit clause. Set `x5=True` and propagate:

```
(~x6)
(x7 OR x10)
(~x7 OR ~x8)
(~x10)
```

Iter 3: `~x8` is a pure literal. Set `x8=False` and remove all clauses containing `~x8`:

```
(~x6)
(x7 OR x10)
(~x10)
```

Iter 3: `x7` is a unit clause. Set `x7=True` and propagate:

```
(~x6)
(~x10)
```

Iter 4: `~x6` is a pure literal. Set `x6=False` and remove all clauses containing `~x6`:

```
(~x10)
```

Iter 4: `~x10` is a unit clause. Set `x10=False` and propagate.

No clauses remain. &varphi; is satisfied and no branching was required.

---

## Exercise 1.2: Resolution

### Exercise 1.2.a

Show how the resolution algorithm can be used to show
that &varphi; &models; &psi;,
where &varphi; and &psi; are the following propositional formulas.

&psi; is the literal `x4` and
&varphi; is the CNF formula that consists of the following clauses:
```
(x1 OR x2 OR x3 OR x4)
(x1 OR x2 OR ~x3 OR x4)
(x1 OR ~x2 OR x3 OR x4)
(x1 OR ~x2 OR ~x3 OR x4)
(~x1 OR x2 OR x3 OR x4)
(~x1 OR x2 OR ~x3 OR x4)
(~x1 OR ~x2 OR x3 OR x4)
(~x1 OR ~x2 OR ~x3 OR x4)
```

**Answer**:

Iter 1: Resolving each pair yields:

```
(x1 OR x2 OR x4)
(x1 OR ~x2 OR x4)
(~x1 OR x2 OR x4)
(~x1 OR ~x2 OR x4)
```

Iter 2: Resolving each pair yields:

```
(x1 OR x4)
(~x1 or x4)
```

Iter 3: Resolving each pair yields:

```
(x4)
```

Thus &varphi; resolves to `x4`.

We prove &varphi; &models; &psi; via contradiction, showing the unsatisfiability of: &varphi; AND  ~&psi;. Clearly `x4 AND ~x4` is unsatisfiable, therefore  &varphi; &models; &psi;.

---

## Exercise 1.3: Encoding 3SAT

## Exercise 1.3.a

The class 3CNF consists of all propositional logic formulas &varphi;
in conjunctive normal form (CNF) where each clause of &varphi;
contains at most three literals.
The problem 3SAT is the problem of deciding whether a given propositional
logic formula &varphi; in 3CNF is satisfiable.

Show how the problem of 3SAT can efficiently be encoded into the
problem of Integer Linear Programming (ILP).
That is, describe an algorithm, that takes less than exponential time,
that takes as input a 3CNF formula &varphi;,
and that produces an integer linear program *P* that has a feasible
solution if and only if &varphi; is satisfiable.

**Answer**:

Algorithm:

- Map every literal xi to a integer variable zi with domain 0 &leq; zi &leq; 1
- Turn every clause into a linear constraint ... > 0,  replacing ~zi with (1 - zi), and binary OR with (+) (E.g., `x1 OR ~x2 OR ~x7` becomes `z1 + (1 - z2) + (1 - z7) > 0`)

The two problems are now equivalent: there's an integer solution to this ILP if and only if there's a boolean solution to the original 3SAT problem.

---

## Exercise 1.4: CSP with binary variables and AllDifferent constraints

### Exercise 1.4.a

Consider the restricted variant of the constraint satisfaction
problem (CSP), where:
- Each variable has domain {0,1}
- The only constraints are *AllDifferent* constraints, i.e., constraints
where the relation consists of all pairs *(v<sub>1</sub>,...,v<sub>n</sub>)* such that
for each *1 &leq; i < j &leq; n* it holds that *v<sub>i</sub> &ne; v<sub>j</sub>*.

Show that there is an efficient (polynomial-time) algorithm that for each
CSP instance *I* that adheres to these restrictions decides whether or not *I*
has a solution&mdash;and if *I* has a solution, it outputs a solution for *I*.

**Answer**: Since each variable has a binary domain, this can be solved using an arc-consistency algorithm such as AC-3 which is runs in linear time with respect to the number of constraints (O(cd^3) = O(8c) = O(c)). Of course, we can add the additional check that any AllDiff constraint with more than 2 variables immediately results in an unsatisfiable problem. After we reduce the search space to 2-consistent, we can search for a solution in time O(n^2d) = O(2n^2) = O(n^2) time.

---

## Exercise 1.5: Encoding MAX2SAT

### Exercise 1.5.a

The class 2CNF consists of all propositional logic formulas &varphi;
in conjunctive normal form (CNF) where each clause of &varphi;
contains at most two literals.
The problem MAX2SAT is the following problem.
You are given as input a propositional logic formula &varphi; in 2CNF.
The task is to find a truth assignment &alpha; to the variables in &varphi;
that satisfies as many clauses of &varphi; as possible,
i.e., such that there is no &alpha;' that satisfies more clauses in &varphi;
than &alpha; does.

Show how to encode the problem MAX2SAT into answer set programming (ASP),
where you are allowed to use optimization statements (e.g., `#maximize { ... }.`)
That is, describe an algorithm, that takes less than exponential time,
that takes as input a 2CNF formula &varphi;,
and that produces an answer set program *P* whose optimal answer sets
correspond exactly to the truth assignments &alpha; that satisfy a maximal
number of clauses of &varphi;.

```clingo
false(X) :- true(neg(X)).
true(X) :- false(neg(X)).
true(X) :- true(pos(X)).
false(X) :- false(pos(X)).

false(X); true(X) :- literal(X).


unsat :- false(X), false(Y), conjunct(X, Y).
unsat :- false(X), true(Y), conjunct(X, Y).
unsat :- true(X), false(Y), conjunct(X, Y).
unsat :- false(X), false(Y), disjunct(X, Y).

:- unsat.

#maximize {1, X : true(X)}.

#show true/1.
#show false/1.
```

When combined with the example:

```
% model ~a ^ b ^ (~c v d)

neg(a).
neg(c).
pos(b).
pos(d).

literal(neg(a)).
literal(pos(b)).
literal(neg(c)).
literal(pos(d)).

disjunct(neg(c), pos(d)).
conjunct(neg(a), pos(b)).
conjunct(pos(b), disjunct(neg(c), pos(d))).
```

Yields:

`true(d) true(b) false(c) false(a)`

---

## Exercise 1.6

### Exercise 1.6.a

Show that the following linear program *P* has a feasible non-integer solution,
and no feasible integer solution.
The program contains the variables *x<sub>1</sub>*, *x<sub>2</sub>* and *x<sub>3</sub>*,
and the following linear inequalities:
```
x1 >= 0
x1 <= 1
x2 >= 0
x2 <= 1
x3 >= 0
x3 <= 1
2*x1 + 3*x2 + 5*x3 <= 6
2*x1 + 3*x2 + 5*x3 >= 6
```

**Answer**:

This program is only satisfied when 2*x1 + 3*x2 + 5*x3 = 6.  Clearly there is no satisfying assignment of x1, x2, and 3 with domains {0, 1}, but there is at least one satisfying non-integer assignment (x1 = 1, x2 = x3 = 0.5).
