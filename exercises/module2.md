# Exercises related to Non-Monotonic Reasoning and Answer Set Programming

These are exercise questions about *non-monotonic reasoning and answer set programming* that you can use
to get a better understanding of the material related to this topic.
The solutions to these exercise questions are not published here, but you are free
to ask about the solutions (e.g., on the [course Canvas page](https://canvas.uva.nl/courses/10768))
if you get stuck and you are free to discuss these exercise questions and their solutions
with other students in the course.

Moreover, these exercise questions are indicative of the type of questions that
you can expect on the final exam.

---

## Exercise 2.1: Strong Equivalence in Answer Set Programming

### Exercise 2.1.a

Consider the following two logic programs *P<sub>1</sub>* and *P<sub>2</sub>*.
Are these strongly equivalent
(see Definition 7.3.10 in [[Van Harmelen, Lifschitz, Porter, 2008]](https://github.com/rdehaan/KRR-course#hokr))?
If so, explain why this is the case.
If not, give a program *P* such
that *P &cup; P<sub>1</sub>*
and *P &cup; P<sub>2</sub>* have different answer sets.

The program *P<sub>1</sub>*:
```
a :- b.
b :- c.
c :- a.
```

The program *P<sub>2</sub>*:
```
a :- b.
b :- a.
a :- c.
c :- a.
```

These are the same programs, while *P<sub>1</sub>* implies `a :- c`, *P<sub>2</sub>* explicitly declares it. Since they are the same program, they will have the same answer sets.

### Exercise 2.1.b

Consider the following two logic programs *P<sub>1</sub>* and *P<sub>2</sub>*.
Are these strongly equivalent
(see Definition 7.3.10 in [[Van Harmelen, Lifschitz, Porter, 2008]](https://github.com/rdehaan/KRR-course#hokr))?
If so, explain why this is the case.
If not, give a program *P* such
that *P &cup; P<sub>1</sub>*
and *P &cup; P<sub>2</sub>* have different answer sets.

The program *P<sub>1</sub>*:
```
b :- a.
b.
```

The program *P<sub>2</sub>*:
```
b :- a.
:- not b.
```

These programs are not equivalent and, therefore, not strongly equivalent.

### Exercise 2.1.c

Consider the following two logic programs *P<sub>1</sub>* and *P<sub>2</sub>*.
Are these strongly equivalent
(see Definition 7.3.10 in [[Van Harmelen, Lifschitz, Porter, 2008]](https://github.com/rdehaan/KRR-course#hokr))?
If so, explain why this is the case.
If not, give a program *P* such
that *P &cup; P<sub>1</sub>*
and *P &cup; P<sub>2</sub>* have different answer sets.

The program *P<sub>1</sub>*:
```
:- a.
:- not a.
```

The program *P<sub>2</sub>*:
```
a :- not b.
b :- not a.
:- a.
:- b.
```

Both programs are unsatisfiable. They have no answer sets and are therefore strongly equivalent.

### Exercise 2.1.d

Let *P<sub>1</sub>* and *P<sub>2</sub>* be two logic programs
that are strongly equivalent
(see Definition 7.3.10 in [[Van Harmelen, Lifschitz, Porter, 2008]](https://github.com/rdehaan/KRR-course#hokr)).
Let *Q<sub>1</sub>* and *Q<sub>2</sub>* also be two logic programs
that are strongly equivalent.

Show that *P<sub>1</sub> &cup; Q<sub>1</sub>*
and *P<sub>2</sub> &cup; Q<sub>2</sub>*
are strongly equivalent.

---
Denoting A(X) as the answer sets of program X:

Since P1 and P2 are strongly equivalent: `A(S U P1) = A(S U P2)` for any logic program S. Also, `A(S U Q1) = S(U Q2)`.

We are asked to show `A(S U P1 U Q1) = A(S U P2 U Q2)`.

```
A(S U P1 U Q1)
= A((S U P1) U Q1) (Associativity of Union)
= A(S U P2 U Q1) (Substitution)
= A(P2 U (S U Q1)) (Commutativity of Union)
= A(P2 U S U Q2) (Substitution)
= A(S U P2 U Q2) (QED)
```

## Exercise 2.2

### Exercise 2.2.a

Give a logic program *P* that has exactly 1024 answer sets.

```
item(1..1024).
1{choice(X): item(X)}1.

#show choice/1.
```

### Exercise 2.2.b

Give a logic program *P* that has exactly 1000 answer sets.

```
item(1..1000).
1{choice(X): item(X)}1.

#show choice/1.
```

### Exercise 2.2.c

Give a default theory *(W,D)* that has exactly 1024 extensions.

You can replicate binary choices for k numbers by defining two rules:

T : p_i / p_i
T : ~p_i / ~p_i

for i = 1..k

There are 1024 possibilities for 10 binary choices (2^10). So we would have twenty total default theories:

T : p_1 / p_1
T : ~p_1 / ~p_1
...
T : p_10 / p_10
T : ~p_10 / ~p_10

### Exercise 2.2.d

Give a default theory *(W,D)* that has exactly 1000 extensions.

For any positive integer N, and positive integers a and b such that a^b = N, we can define a*b default theories with exactly N extensions.

Each rule_i,j is defined as T: (~p_i_j, ... for all i, j such that i &ne; j) / p_i_j for i=1..a, j=1..b

In the case N=1000, then a=3, b=10:

So we have rules:

T : ~x_1_2, ... , ~x_1_10 / x_1_1
....
T: ~x_3_1, ... , ~x_3_9 / x_3_10

In Answer Set Programming the rules becomes:

```
x_1_1 :- not x_1_2, ..., x_1_10.
...
x_3_1 :- not x_3_1, ..., x_3_9.
```

---

## Exercise 2.3: Modelling non-monotonic reasoning

### Exercise 2.3.a

Model the following (made-up!) scenario of legal reasoning using default logic:
- A suspect should be convicted if they committed a crime.
- Theft is a crime.
- Murder is a crime.
- A suspect should not be convicted if they have immunity from prosecution.
- Immunity from prosecution does not hold if the suspect committed a murder.

To keep things simple (or to oversimplify things),
use the following propositional atoms for this:
`convicted`, `committed_crime`, `committed_theft`, `committed_murder`, `immunity`.
(You may use additional atoms.)

That is, construct a set *D* of defaults, so that:
- Together with *W = {`committed_theft`}*, all extensions of the resulting default
theory contain `convicted`.
- Together with *W = {`committed_theft`,`immunity`}*, all extensions of the resulting default
theory do not contain `convicted`.
- Together with *W = {`committed_murder`}*, all extensions of the resulting default
theory contain `convicted`.
- Together with *W = {`committed_murder`,`immunity`}*, all extensions of the resulting default
theory contain `convicted`.

```
D = {
    committed_murder : T / committed_crime
    committed_theft : T / committed_crime
    immunity : ~committed_murder / exception
    committed_crime : ~exception / convicted
}
```

### Exercise 2.3.b

Model the above scenario of legal reasoning using answer set programming.
Use the same propositional atoms (again, you may use additional atoms).

That is, construct a logic program *P*, so that:
- All answer sets of *P &cup; {`committed_theft.`}* contain `convicted`.
- All answer sets of *P &cup; {`committed_theft.`,`immunity.`}* do not contain `convicted`.
- All answer sets of *P &cup; {`committed_murder.`}* contain `convicted`.
- All answer sets of *P &cup; {`committed_murder.`,`immunity.`}* contain `convicted`.

2.3.a directly translated to ASP:
```
commited_crime :- committed_theft
committed_crime :- commited_murder
exception :- immunity, not commited_murder
convicted :- commited_crime, not exception
```

### Exercise 2.3.c

Model the following (made-up!) scenario of deontic reasoning using default logic:
- One should not do things that are forbidden.
- Speeding is forbidden.
- Fraud is forbidden.
- Forbidden things may be done if they can save lives.
- In an emergency, speeding can save lives.

To keep things simple (or to oversimplify things),
use the following propositional atoms for this:
`should_not_do(speeding)`, `allowed_to_do(speeding)`,
`should_not_do(fraud)`, `allowed_to_do(fraud)`,
`forbidden(speeding)`, `forbidden(fraud)`,
`can_save_lives(speeding)`, `emergency`.
(You may use additional atoms.)

That is, construct a set *D* of defaults
and a set *W<sub>0</sub>* of background facts, so that:
- Together with *W = W<sub>0</sub> &cup; {`emergency`}*,
all extensions of the resulting default
theory contain `forbidden(speeding)`, `forbidden(fraud)`,
`allowed_to_do(speeding)` and `should_not_do(fraud)` and do not
contain `allowed_to_do(fraud)` and `should_not_do(speeding)`.
- Together with *W = W<sub>0</sub> &cup; &emptyset;*,
all extensions of the resulting default
theory contain `forbidden(speeding)`, `forbidden(fraud)`,
`should_not_do(speeding)` and `should_not_do(fraud)` and do not
contain `allowed_to_do(fraud)` and `allowed_to_do(speeding)`.

### Exercise 2.3.d

Model the above scenario of deontic reasoning using answer set programming.
Use the same propositional atoms (again, you may use additional atoms).

That is, construct a logic program *P*, so that:
- All answer sets of *P*
contain `forbidden(speeding)`, `forbidden(fraud)`,
`allowed_to_do(speeding)` and `should_not_do(fraud)` and do not
contain `allowed_to_do(fraud)` and `should_not_do(speeding)`.
- All answer sets of *P &cup; {`emergency.`}*
contain `forbidden(speeding)`, `forbidden(fraud)`,
`should_not_do(speeding)` and `should_not_do(fraud)` and do not
contain `allowed_to_do(fraud)` and `allowed_to_do(speeding)`.

```
forbidden(speeding;fraud).
can_save_lives(speeding).
allowed_to_do(X) :- emergency, can_save_lives(X).
should_not_do(X) :- forbidden(X), not allowed_to_do(X).
```
---

## Exercise 2.4

### Exercise 2.4.a

Show that the default theory *(W,D)* with *W = &emptyset;*
and *D = { `T : p / ~q`, `T : q / ~r`, `T : r / ~s` }*
(where `~` denotes negation)
has exactly one extension.

This corresponds to the following logic program in ASP:

```
p_not_q :- not p_not_p.
p_not_r :- not p_not_q.
p_not_s :- not p_not_r.
```

Which yields exactly one answer set (extension): `~q, ~s`

### Exercise 2.4.b

What extensions does the default theory *(W,D)* with *W = { `p -> (~q & ~r)` }*
and *D = { `T : p / p`, `T : q / q`, `T : r / r` }*
(where `~` denotes negation, `&` denotes conjunction, and `->`
denotes logical implication) have?

In ASP:
```
p_p :- not p_not_p.
p_q :- not p_not_q.
p_r :- not p_not_r.
p_not_q, p_not_r :- p_p.
```

Which yields two answer sets (extensions): `p, ~q, r` and `p, ~r, q`

### Exercise 2.4.c

Show that the default theory *(W,D)* with *W = { `p` }*
and *D = { `p : r / q`, `p : s / ~q` }*
(where `~` denotes negation)
has no extensions.

In ASP:

```
:- literal(X), not_literal(X).

literal(p).

literal(q) :- literal(p), not not_literal(r).

not_literal(q) :- literal(p), not not_literal(s).
```

This program is unsatisfiable and therefore has no extensions.
