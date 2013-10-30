# We should be able to:
# 1) use uv-data in different interferometric standards => internal
# presentation of data should be smth like numpy ndarrays or recarrays.
# 2) substitute data on model at any baseline(s) => class Model
# 3) use gain curves as from uv-data files (AIPS AN tables) or in any other
# format => class Gains
# 4) Bootstrapp data using resuduals between data and model multplied on final
# set of gains. Using parametric or nonparametric bootstrapp. Spatial in case
# of baselines with different sensetivities.
# 5) Preparing testing and training samples from data file for k*r CV.
# 6) Given model (instance of Model class) and data (instance of Data class)
# calculate lnLikelihood for data given model parameters. Model should contain
# interfaces for changing parameters.
