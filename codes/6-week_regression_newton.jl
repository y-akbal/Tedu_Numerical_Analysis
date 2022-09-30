using Zygote
using ForwardDiff
using Plots
using LinearAlgebra
using Distributions


X = rand(Uniform(-100,100),(100,5))
coeff = randn(5,1)
bias = randn()
y = X*coeff .+ bias   ####This is our data set over all we have 100 samples with 5 dimensions

#= we will do the same Newton's method=#
@info("This is pretty usefull macro")

