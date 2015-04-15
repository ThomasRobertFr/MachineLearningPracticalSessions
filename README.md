# Machine Learning Practical Sessions

This repository gather almost all Machine Learning practical sessions I attended during my studies at the INSA of Rouen & the University of Rouen in 2013 and 2014.

For most of theses sessions, the point was to implement Machine Learning algorithms to get a sense of how they work.

Also, having been written in French schools, most of the comments are in French, although I sometimes wrote them in English. Sorry for non-French speakers. However, code and charts should be understandable.

## Organization

Each practical session consist of a folder containing the sources (that you should be able to run) and a report of the results as a PDF file.

Each session is numbered `x.y`, where `x` correspond to the number of the course, and `y` correspond to the number of the practical session of the course. I numbered the courses approximately chronologically.

## About the code

The code is either in Matlab (for the most part) or in Python (for a few ones).

### Matlab

For each session, the main file that should be run is usually prefixed by an underscore.

For Matlab, the following libraries might be needed:

* [CVX](http://cvxr.com/cvx/) (very often)
* [SimpleMKL](http://asi.insa-rouen.fr/enseignants/~arakoto/code/mklindex.html) (very often for `monqp.m`)
* [PROPACK](http://sun.stanford.edu/~rmunk/PROPACK/) (rarely)
* [PRTools](http://prtools.org/) (very rarely)
* [SOM Toolbox](http://www.cis.hut.fi/somtoolbox/) (very rarely)

### Python

For Python you will need the `scipy` environment, `pickle`, `pykalman` and `yahmm`.

The code is available as a Python file or a IPython Notebook.
