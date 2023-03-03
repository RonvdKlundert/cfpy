
## To Do

- [x] add normalization CF
- [x] remove Subsurface functionality from this repo, add to [dncf](https://github.com/RonvdKlundert/cfdn) so this repo can lose the pycortex depency
- [] add grid fit for normalization CF
- [] ..


## cfpy

cfpy is a package that allows you to simulate 
and fit population receptive field (pRF) parameters from time series data
and fit connective field (CF) parameters from time series data.


To get started using these components in your own software, clone this repository, then run

python installer.py

this will try to install dependencies with conda, and if not possible, with pip. The only
required dependencies are:

numpy>=1.16
scipy>=1.4
statsmodels>=0.8
joblib
nilearn

To install dependencies with pip directly, run

python -m pip install -e .


## License

``prfpy`` is licensed under the terms of the GPL-v3 license. See the file
"LICENSE" for information on the history of this software, terms & conditions
for usage, and a DISCLAIMER OF ALL WARRANTIES.

All trademarks referenced herein are property of their respective holders.

Copyright (c) 2022--, Marco Aqil, 
Spinoza Centre for Neuroimaging, Amsterdam.
