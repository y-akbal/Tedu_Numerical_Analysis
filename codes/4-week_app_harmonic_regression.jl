using Zygote
using Plots

### Syntetic data we create below
A = -15:105
B = [cos(i)+2*sin(i/3)for i in A]
plot(B)




A = Float32.(A)
B = Float32.(B)

### Below is the linearregression object
Base.@kwdef mutable struct HarRegression
a::Vector{Float32}
b::Vector{Float32}
end


###instantiate it!
lr = HarRegression(randn(25), rand(25))


(lr::HarRegression)(x) = sum([lr.a[i+1]*cos(x/i)+lr.b[i+1]*sin(x/i) for i in 1:length(lr.a)-1])+lr.a[1]+lr.b[1]


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
    lr.a -= 0.0000015f0*grad[:a]
    lr.b -= 0.0000015f0*grad[:b]
    if i%100 == 0
        println(loss(lr.(A),B))
    end
end    

plot(B)
plot!(lr.(A))
