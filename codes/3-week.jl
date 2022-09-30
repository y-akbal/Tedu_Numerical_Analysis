#=
Today's Dish:
1) Regression Problems (Interpolation)
2) Harmonic Regression
=#
using Pkg
using DataFrames
using CSV
using Statistics
using Plots
using LinearAlgebra

csv_reader = CSV.File("ad.csv")
data = DataFrame(csv_reader)  #### created a data frame here

X = data[:,1]
y = data[:, 2]

scatter(X,y)   #### there is a linear correlation surely
cor(X,y)   ### as you see correlation is pretty high!

#= Let's implement our own linear regression thing =#
Base.@kwdef mutable struct LR
    a:: Float64
    b:: Float64
end

lr = LR(1,1) ##intitialize the linear regression object!
(lr::LR)(x::Float64) = lr.a * x + lr.b  ## evaluation procedure is here now!
(lr::LR)(x::Array) = lr.(x)  ### it is for forward pass everything


function fit!(lr::LR, X::Vector{Float64}, y::Vector{Float64})::Nothing
    mx = mean(X)
    my = mean(y)
    x_0 = X .- mx
    y_0 = y .- my
    x_0_s = sum((x_0).^2)
    lr.a = transpose(x_0)*y_0/x_0_s
    lr.b = my - lr.a*mx
    println("Model fitted succesfully!")
end

fit!(lr, X, y)

###Let's sketch the predictions
scatter(X, y, label = "Real")
plot!(X, lr.(X), label = "Predicted")


function mse(lr::LR, X::Vector{Float64}, y::Vector{Float64})
    return sum((lr(X) - y).^2)
end

mse(lr, X, y)


function R2(lr::LR, X::Vector{Float64}, y::Vector{Float64})
    return 1 - sum((lr(X) - y).^2)/sum((y .- mean(y)).^2)
end

R2(lr, X, y)

###Let's visualize the residuals
plot(X, (lr(X) - y).^2)