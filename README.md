# Master Thesis
## Automatic Differentiation for Environmental Models

The thesis has two scenarios for the application of the automatic differentiation(AD) in Environmental Models(EMs).

1) AD for model calibration: AD, forward AD and backward AD, has been mainly compared with finite difference method for gradient calculation. Two models have been used, the scalable linear regression model and the scalable ode model. The code is in the file "Param_Est", and the packages used are listed in Manifest.toml and Project.toml files.

2) AD + NN: to use neural network to replace part of the model. There are two models used in this thesis: the toy model (as named "toymodel_main.jl" and the russikon conceptual model (as named "batch_main.jl"). There is also sensitivity analysis for the original conceptual model. The code is in the file "ODE+NN", and the packages used are listed in Manifest.toml and Project.toml files.

The report of the thesis can be seen from:
https://www.overleaf.com/read/rztwzbrsmtwm
