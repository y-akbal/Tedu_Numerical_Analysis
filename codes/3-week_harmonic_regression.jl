using Zygote
using Plots

g(x) = x - floor(x)-1/2+sin(2*pi*x)
B = [g(i) for i in 1:0.1:1000]
scatter(B, markersize=3,alpha=.9,legend= false, label = "Sample")
scatter!(xlims=(1,500))

A = Float32.(1:0.1:1000)
B = Float32.(B)
### Below is the linearregression object
Base.@kwdef mutable struct HarRegression
a::Vector{Float32}
b::Vector{Float32}
c::Vector{Float32}
end

###instantiate it!
lr = HarRegression(randn(Float32,10), randn(Float32, 10),Float32.(zeros(1))) 
pi_ = Float32(pi)


@inline function (lr::HarRegression)(x::Float32)
    sum = lr.c[1]   
    for (i,(a,b)) in enumerate(zip(lr.a, lr.b))
        sum += a*sin(2*pi_*i*x)+ b*cos(2*pi_*i*x)
    end
    return sum
end
#### Calculate individual losses
function ind_loss(x::Float32, y::Float32)
        return (x - y)^2
end        

### Loss over all dataset
function loss(x::Vector{Float32}, y::Vector{Float32})
    return sum((x-y).^2)
end



using LinearAlgebra
@time for i in 1:50000
     
    grad = gradient(lr->loss(lr.(A[1:100]),B[1:100]), lr)[1]
    lr.a -= 0.0015f0*grad[:a]/norm(grad)
    lr.b -= 0.0015f0*grad[:b]/norm(grad)
    lr.c -= 0.0015f0*grad[:c]    
    #lr.L -= 0.00015f0*grad[:L]
    if i%100 == 0
        println(loss(lr.(A),B),"\t")
    end
end    


scatter(B, markersize=3,alpha=.9, legend = true, label = "Sample")
scatter!(xlims=(100,300))

plot!(lr.(A), markersize=1,alpha=.9, legend = true, label = "Fitted_Curve")
plot!(xlims=(100,300))

plot!(lr.(A))
