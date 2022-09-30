using Zygote
using LinearAlgebra
using Plots
using ForwardDiff

Q = [1 0 0;0 3 0; 0 0 9]

b = randn(3)
f(x::Vector{Float64}) = transpose(x)*Q*x -transpose(b)*x


x = rand(3)
function grad_descent(f, x; n_iter = 100, lr = 0.004)
    for i in 1:n_iter
        grad = Zygote.gradient(f, x)[1]
        x -= lr*grad
    end
    return x
end

grad_descent(f, x, n_iter = 10000)



function fit!(f::Function; n_iter = 100::Int64)
    x = randn(3)
    h = 0.0001
    for i in 1:n_iter
        grad = Zygote.gradient(x -> f(x), x)[1]
        phi(α,f::Function) = f(x - α*grad)
        D(α) = (-phi(α-h,f) + phi(α+h,f))/(2*h)  ### we compute the derivative here approximately bacause it saves some time!
        α_ = find_zero(α -> D(α), (0,150))
        x -= α_*grad
    end
    return x
end

fit!(f, n_iter = 50)

