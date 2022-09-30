using Zygote

### Syntetic data we create below
A = randn(100)
B = 2*A .+1
A = Float32.(A)
B = Float32.(B)

### Below is the linearregression object
Base.@kwdef mutable struct LinearRegression
a::Float32
b::Float32
end


###instantiate it!
lr = LinearRegression(randn(), randn())

(lr::LinearRegression)(x) = lr.a*x+lr.b


#### Calculate individual losses
function ind_loss(x::Float32, y::Float32)
        return (x - y)^2
end        

### Loss over all dataset
function loss(x::Vector{Float32}, y::Vector{Float32})
    return sum((x-y).^2)
end


using ForwardDiff

for _ in 1:20000
    grad = gradient(lr-> loss(lr.(A),B), lr)[1]
    lr.a -= 0.000015f0*grad[:a]
    lr.b -= 0.000015f0*grad[:b]
    println(loss(lr.(A),B))
end    

