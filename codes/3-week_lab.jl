using LinearAlgebra
using Statistics
### we use permutedims for finding transpose of a matrix
### we create some syntetic data
X = sort(rand(1000))
X = Array(X)
f(x) = (x-0.5)*(x-0.3)*(x-0.8)
y = f.(X)

#=
using DataFrames
using CSV
csv_reader = CSV.File("ad.csv")
data = DataFrame(csv_reader)  #### created a data frame here
=#


using Plots
scatter(X, y) ### the data looks pretty scary!
## create an instance that will store the coefficients of the polynomial
Base.@kwdef mutable struct pol
    degree::Int
    coeff::Vector{Float64}
    fitted::Bool = false ### this is important because we do not want to change this later.
end


function pol(x::Int64) ## enjoy multiple dispatch, we are constructing the pol object here
    degree = x
    coeff = randn(degree+1)
    return pol(degree = degree, coeff = coeff)
end

p = pol(60)

function (p::pol)(x)  ###the object works like a function now (forwad pass)
    local s = p.coeff[1]
    for (i,j) in enumerate(p.coeff[2:end])
        s += j*x^i  #Syntactic sugar :)
    end
    return s
end

function fit!(p::pol, X::Vector{Float64}, y::Vector{Float64})  ###if we are to change p, convention is to use fit! instead of fit
        p.fitted ? throw("The model is alrady fitted!") : p.fitted = true
        
        B_p = reduce(hcat, [X.^i for (i,_) in enumerate(p.coeff[2:end])])
        B = transpose(hcat(ones(length(X)), B_p))
        A = B*transpose(B)
        y_ = B*y
        coeff_ = A\y_
        p.coeff = coeff_
        error = (1/length(X))*sum((p.(X)-y).^2)
        println("Fitting is done, the error is $error")
end   

fit!(p, X, y)


@show p.coeff


using Plots
scatter(X, y, label = "data") ### the data looks pretty scary!
plot!(X, p.(X), label = "fitted_curve")
## create an instance that will store the coefficients of the polynomial


function R2(p::pol, X::Vector{Float64}, y::Vector{Float64})
    return 1 - sum((p.(X) - y).^2)/sum((y .- mean(y)).^2)
end

R2(p, X, y)

### Let's find the best degree
L = []
for i in 1:150
    p = pol(i)
    fit!(p, X, y)
    push!(L, R2(p, X, y))
    println(i)
end

p = pol(argmax(L))
fit!(p, X, y)
p(8.5)  ##dolares!
