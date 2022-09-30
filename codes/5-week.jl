### Gradient Descent example

using Pkg
using ForwardDiff
using Plots
using BenchmarkTools
f(x) = x[1]^2 +10*x[1] + x[2]^2 - 20*x[2] + 10
∇f(x) = [2*x[1]+10, 2*x[2]- 20]

x = rand(2)
Lx = []
Ly = []
 for i in 1:1000
    push!(Lx, x[1])
    push!(Ly, x[2])
    ###x -= 5*ForwardDiff.gradient(f, x)
    x -= 0.05*∇f(x)
    x 
    loss = f(x)
    print("$loss, $i, The point is: $x \n")
end

scatter(Lx, Ly)


using Optim

result = optimize(f, zeros(2), BFGS())