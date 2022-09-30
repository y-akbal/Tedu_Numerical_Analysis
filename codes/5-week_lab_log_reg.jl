using Distributions
using LinearAlgebra
using Plots
using Zygote
using ForwardDiff

a = randn()
b = randn()
d = randn()

f(x) = a*x[1]+ b*x[2]+ d
g(x) = -( d +a*x)/b ### decision boundary function


function f(x::Matrix{Float64})   ### we are overloading the function
    return [f(x[i,:]) for i in 1:size(x)[1]]
end



function return_syntetic_data(n::Int, f::Function, dim = 2::Int)
    X = zeros(n, dim)
    y =  Float64.([])
    for i in 1:n
        x = rand(Uniform(-3,3), dim)
        if f(x) > 0
            X[i,:] = x
            push!(y, 1)

        elseif f(x) <= 0
            X[i,:] = x
            push!(y, 0)
        end
    end
    return X, Float64.(y)
end

(X, y) = return_syntetic_data(100, f)

pos_ind = [i for (i,y_) in enumerate(y) if y_ == 1.0]
neg_ind = [i for (i,y_) in enumerate(y) if y_ == 0.0]

X_neg = X[neg_ind,:]
X_pos = X[pos_ind, :]

l = minimum(X[:,1])
m = maximum(X[:,1])

scatter(X[pos_ind,1], X[pos_ind,2], color ="red", label = "1")
scatter!(X[neg_ind,1],X[neg_ind,2], color ="blue", label = "0")
plot!(l:0.1:m, g.(l:0.1:m), color = "black", label = "dec")



[f(X_neg[i,:]) for i in  1:size(X_neg[:,:])[1]]  ### these guys are all negative
[f(X_pos[i,:]) for i in  1:size(X_pos[:,:])[1]]  ### these guys are all negative


function sigmoid(x::Real; a = 1::Int64)
    return 1/(1+exp(-a*x))
end


Base.@kwdef mutable struct LR
    θ::Vector{Float64}
    adjusted_ = false :: Bool
end

function LR()  ### for initialization
    return LR(θ = randn(3), adjusted_ = false)
end

lr = LR()


function accuracy(y_logits, y; cutoff = 0.5 :: Float64)
    @assert length(y_logits) == length(y)
    y_ = [i < cutoff ? 0 : 1 for i in y_logits]
    L = [i == j for (i,j) in zip(y_, y)]
    return sum(L)/length(L)
end



function (lr::LR)(X::Matrix{Float64})   ####Used for forward pass 
    if lr.adjusted_
        X = hcat(X, ones(size(X)[1]))
        return sigmoid.(X*lr.θ)  ### we apply broadcasting
    else
        lr.θ = 0.5*randn(size(X)[2]+1)
        lr.adjusted_ = true
        return lr(X)
    end
end


function (lr::LR)(X::Vector{Float64})   ####Used for forward pass for a single element
    if lr.adjusted_
        L = vcat(X, 1)
        return sigmoid(dot(L,lr.θ))  ### we apply broadcasting
    else
        lr.θ = 0.5*randn(size(X)[2]+1)
        lr.adjusted_ = true
        return lr(X)
    end
end




function loss_cross_entropy(y_pred, y_test)
    A = y_test.*(log.(y_pred)) + (1 .- y_test).*(log.(1 .-y_pred))
    return -mean(A)
end

function fit!(lr, X, y; max_iter = 1000, lr_ = 0.005)
    for i in 1:max_iter
        loss  = loss_cross_entropy(lr(X), y)
        accuracy_ = accuracy(lr(X), y)
        lr.θ -= lr_*Zygote.gradient(lr -> loss_cross_entropy(lr(X), y), lr)[1][1]
        if i % 100 == 0
        print("loss is $loss, accuracy is $accuracy_ \n")
        end
    end
end
lr = LR()
fit!(lr, X,y, max_iter = 10000, lr_ = 0.005)

t(x) = -(lr.θ[1]*x + lr.θ[3])/lr.θ[2]



#=  We are sketching the decision boundary of log_regression =#

scatter(X[pos_ind,1], X[pos_ind,2], color ="red", label = "1")
scatter!(X[neg_ind,1],X[neg_ind,2], color ="blue", label = "0")
plot!(l:0.1:m, g.(l:0.1:m), color = "black", label = "dec")
plot!(l:0.1:m, t.(l:0.1:m), color = "purple", label = "log_dec")

