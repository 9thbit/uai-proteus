# UAI-Proteus

This repository constitutes the Proteus submission to the UAI 2014 Inference
Competition (MPE category) which achieved first place under both the 20 and 60
minute timouts, and second place for the 20 second timeout.

The portfolio aims to benefit from complimentary performances among solvers on
various types of instances. By intelligently choosing the solver on a
per-instance basis, significant performance gains can be achieved overall. This
portfolio is composed of 5 component solvers:

* Toulbar2  
  Soft arc consistency revisited  
  Martin C Cooper, Simon de Givry, Marti Sánchez, Thomas Schiex, Matthias Zytnicki, T Werner  
  Artificial Intelligence 174 (7), 449-478  

* Toulbar2 preceeded by incop  
  Bertrand Neveu, Gilles Trombettoni, Fred Glover  
  ID Walk: A Candidate List Strategy with a Simple Diversification Device  
  Principles and Practice of Constraint Programming -- CP 2004  

* mplp2  
  David Sontag, Do Kook Choe, Yitao Li  
  Efficiently Searching for Frustrated Cycles in MAP Inference  
  Uncertainty in Artificial Intelligence -- UAI 2012  

* Two encodings, direct and tuple, to ILP and solved using CPLEX.  
  The encodings and interface where implemented by George Katsileros, INRA.  
  IBM ILOG CPLEX http://www-01.ibm.com/software/commerce/optimization/cplex-optimizer/  

The interface to CPLEX is implemented directly in uai-proteus.cc, but other
solvers are launched as external processes.


## UAI Training Instances

Performance data of each solver was collected on a collection of 2556 instances
which have been encoded in UAI format from fields such as MRF, CVPR, CFN, and
Max CSP by INRA: [http://genoweb.toulouse.inra.fr/~degivry/evalgm/]
We thank the GenoToul Bioinformatics Platform of INRA-Toulouse for providing
computing resources and support to collect this data.


## CSV data

* feattimes.csv lists the time needed to compute features of the instances.
* features.csv is the feature matrix
* objectives.csv is a matrix containing the best objective obtained by each
  solver within the time limit.
* success.csv is a matrix with a 0/1 for each solver indicating if that solver
  produced a valid solution on an instance.
* times.csv is a matrix of the time used by each solver on each instance
* scores.csv is computed from objectives and times.csv by `computescore.py`


## Requirements

### To compile the Proteus portfolio

* C++x11 compiler
* Boost
* IBM ILOG CPLEX 12.5.* or 12.6


### To retrain the prediction models (optional)

* sklearn & numpy
* Optionally `pydot` if you would like to plot the decision trees.

