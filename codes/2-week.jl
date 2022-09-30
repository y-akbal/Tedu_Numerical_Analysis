using Printf



function root_find(f::Function, a::Number, b::Number, ϵ= 1e-5::Float64)
    @assert f(a)*f(b) < 0  ### we assert here whether f has at least one zero in (a,b)
    c = 0.0::Float64  ### because of the last return c, without this an error is thrown.
    while abs(b-a) > ϵ
        c = (a+b)/2
        
        if f(a)f(c) < 0
            b = c
        elseif f(c)*f(b) < 0
            a = c
        elseif f(c) == 0
            return c
        end
    end
    return c
end


a = root_find(x -> sin(x), 2, 4, 1e-10)
@printf("%0.27f", a) 

f(x) = x^2 -2

t = root_find(f, 0, 2, 1e-10)


using ForwardDiff 
"""
For the sake of brevity we use automatic differentiation library, otherwise we gotta evaluate the derivative on our own
"""

function newton_raphson(f::Function; n_its = 100, initial_point = "random"::Union{string, Float64})
    if initial_point == "random"
        x = randn() # initialize a random point
    else
        x = initial_point
    end
    for _ in 1:n_its
        x = x - f(x)/ForwardDiff.derivative(f,x)
    end
    return x
end


newton_raphson(x -> x^2 - 10, initial_point = 2)  ###this macro allows one to see time to run the function


function secant_method(f::Function; n_its = 100)
    x = randn()
    x_ = randn()
    for _ in 1:n_its
        x,x_ = x - (x-x_)*f(x)/(f(x)- f(x_)+1e-10), x   ####The correction factor dude is introduced here
    end
    return x
end

secant_method(x->sin(x), n_its = 10000000)