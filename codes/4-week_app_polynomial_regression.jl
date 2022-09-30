using Zygote
using Plots

### Syntetic data we create below
A = -5:0.1:5
B = sum([randn()*A.^i for i in 0:5])
scatter(A, B)

A = Float32.(A)
B = Float32.(B)

### Below is the linearregression object
Base.@kwdef mutable struct PolRegression
a::Vector{Float32}
end


###instantiate it!
lr = PolRegression(randn(7))


(lr::PolRegression)(x) = sum([lr.a[i+1]*x^i for i in 1:length(lr.a)-1])+lr.a[1]


#### Calculate individual losses
function ind_loss(x::Float32, y::Float32)
        return (x - y)^2
end        

### Loss over all dataset
@inline function loss(x::Vector{Float32}, y::Vector{Float32})
    return sum((x-y).^2)
end



using LinearAlgebra
@time for i in 1:50000
    grad = gradient(lr-> loss(lr.(A),B), lr)[1]
    lr.a -= 0.0015f0*grad[:a]/(norm(grad)+1e-4)+1/i*randn(length(lr.a))
    if i%100 == 0
        println(loss(lr.(A),B))
    end
end    

plot(B)
plot!(lr.(A))
