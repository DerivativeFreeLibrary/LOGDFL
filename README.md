# Algorithm LOG-DFL

A penaty-barrier derivative-free algorithm for the solution of constrained black-box minimization problems.
To use Algorithm LOG-DFL onto a user-defined problem:

* edit file ```problems.py``` to add the user-defined problem (following the three provided example problems). Add the newly added class name to the ```names``` list at line 3

* modify file ```LOG-DFL.py``` to optimize the user-defined problem (e.g. at line 855 put the name of the user-defined problem to be solved).
Remember to change (if needed):
  * ```maxfun```
  * ```tol```
  * ```iprint```

## To be noted
LOG-DFL shall handle with an interior penalty approach those inequality constraints that are <b>strictly</b> satisfied at the initial given point. All the other constraints are automatically handled by a sequential exterior penalty approach. There is no need to split equality constraints into two inequalities.

### Authors
A. Brilli, G. Liuzzi, S. Lucidi

#### Contacts
* Andrea Brilli: `brilli@diag.uniroma1.it`
* Giampaolo Liuzzi: `liuzzi@diag.uniroma1.it`
* Stefano Lucidi: `lucidi@diag.uniroma1.it`

#### Copyright (2022)
