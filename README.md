# A Mass Conserving Mixed Stress-Strain rate Finite Element Method for Non-Newtonian Fluid Simulations. 
This repository contains all the necessary codes to produce the results
presented in the thesis
### Prerequisites:
- Finite element software NGSolve/Netgen (>= 6.2.2102) with Intel MKL Pardiso
- Python packages: numpy, matplotlib, pandas

### Usage: 
The script `Setup.py` contains all the parameters and can be seen as `main.cpp`.
Some parameters (geometry, output options etc..) have to be set explicitly in the script, while some
other parameters can be passed via terminal. These are:

- `-struct` &#8594; Creates a structured mesh (works only on rectangular domains)
- `-nn`     &#8594; Select non-Newtonian model (binary)
  - `1` ... Powerlaw
  - `2` ... Bingham
  - `3` ... Powerlaw + Bingham
- `-hmax`   &#8594; Pass mesh size (multiple values)
- `-kappa`  &#8594; Pass regularisation parameter (multiple values)
- `-thr`    &#8594; Define number of threads

In addtion to the upper defined optional arguments, there exist
three compulsory arguments:

- `bash`      &#8594; Executes simulation with `python3` or `netgen`
- `benchmark` &#8594; Choose between `unit`,`periodic`,`cavity` or `cylinder`.
  - `unit`              ... Performs Newtonian testing on unit square
  - `periodic`          ... Non-Newtonian simulation in a periodic channel
  - `cavity`            ... Non-Newtonian simulation in lid driven cavity
  - `cylinder`          ... Non-Newtonian simulation flow around cylinder
- `fem`       &#8594; Finite element (binary)
  - `1` ... MCS
  - `2` ... MCS-S
  - `3` ... MCS + MCS-S
  - `4` ... TH-S
  - `5` ... TH-S + MCS
  - `8` ... SV-S
  - etc...

For example: `python3 setup.py netgen periodic 5 -nn 2 -hmax 0.1 0.05 -kappa 1e-4 1e-7 -struct` \
exhibits a Non-Newtonian Bingham simulation in a periodic channel with a structured mesh of size 0.1 and 0.5 
and two regularisation parameters kappa each for the Taylor-Hood+Stress and the MCS element. Thus eight simulations in total

Features, Non-Newtonian model or Finite Element can be integrated rather fast. 

