using Zygote
using ForwardDiff
using Plots
using LinearAlgebra

function get_grad(x::Vector{Float64}, f::Function)
    return Zygote.gradient(t -> f(t), x)[1]
end

function get_hessian(x::Vector{Float64}, f::Function)
    return Zygote.hessian(t -> f(t) ,x)
end

Q = 2*randn(3,3)
Q = transpose(Q)*transpose(Q)
b = randn(3)


k(x) = (1/2)-transpose(x)*Q*x -transpose(b)*x
x = rand(3)


function fit!(f, x; max_iter = 100)
    for i in 1:max_iter
        x -= (get_hessian(x,k)^(-1))*get_grad(x,k)
    end
    if det(get_hessian(x,k)) > 0
        println("Algorithm Converged the minimum is $(f(x))")
        global x = x
    else
        println("Algorithm did not converge!")
    end
    
end

x = 10000*randn(3)   
fit!(f,x, max_iter = 10)
println(x)
