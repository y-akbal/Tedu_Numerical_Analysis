using Interpolations, Plots

# Lower and higher bound of interval
a = 1.0
b = 20.0
# Interval definition
x = a:1.0:b
# This can be any sort of array data, as long as
# length(x) == length(y)
y = @. cos(x^2 / 9.0) # Function application by broadcasting
# Interpolations
itp_linear = linear_interpolation(x, y)
itp_cubic = cubic_spline_interpolation(x, y)
itp_quadratic = Quadratic(x,y)
# Interpolation functions
f_linear(x) = itp_linear(x)
f_cubic(x) = itp_cubic(x)
f_quadratic(x) = itp_quadratic(x)
# Plots
width, height = 1500, 800 # not strictly necessary
x_new = a:0.1:b # smoother interval, necessary for cubic spline

scatter(x, y, markersize=10,label="Data points")
plot!(f_linear, x_new, w=3,label="Linear interpolation")
plot!(f_cubic, x_new, linestyle=:dash, w=3, label="Cubic Spline interpolation")
plot!(size = (width, height))
plot!(legend = :bottomleft)



"""
Numerical Derivative Part
"""
#### One sided derivatives

function one_sided_p(f::Function, x; h = 1e-10)
        return (f(x+h) - f(x))/h
end
function one_sided_n(f::Function, x; h = 1e-10)
    return -(f(x-h) - f(x))/h        
end


function double_sided(f::Function, x; h = 1e-10)
    return (f(x+h) - f(x-h))/(2h)
end

f(x) = (x^3)*sin(x) + (x^10)*cos(x)

using ForwardDiff
double_sided(f, 5)
one_sided_p(f, 5)
one_sided_n(f,5)

ForwardDiff.derivative(f, 5.0)  #### actual values (the one as close as possible)

using ForwardDiff, Zygote
using BenchmarkTools



function local_ext_search(f::Function;n_iterations = 100::Int64)
    x = randn()
    f_d(x) = ForwardDiff.derivative(f,x)
    f_dd(x) = ForwardDiff.derivative(f_d,x)
    for _ in 1:n_iterations
        x -= (f_d(x)+1e-100)/(f_dd(x)+1e-10)   ####Syntactic sugar :)
    end
    return x
end

f(x) = sin(x)/(1+x^2)

local_ext_search(f;n_iterations = 10000000)