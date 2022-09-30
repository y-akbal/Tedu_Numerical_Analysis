using Zygote
using ForwardDiff
using Plots
using LinearAlgebra
using Distributions
using Progressbars


X = rand(Uniform(-100,100),(100,5))
coeff = randn(6,1)

y = X*coeff[1:5,:] .+ coeff[6,:]   ####This is our data set over all we have 100 samples with 5 dimensions

function forward(X, coeff)
    l = ones(size(X)[1])
    X_ = [X l]
    return X_*coeff
end



coeff_i = randn(6,1) #### This dude is our random coefficients, 

loss(y, y_) = (1/length(y))*sum((y-y_).^2) #### this dude is to be differentiated.
loss(forward(X, coeff_i), y)   #### let's what the loss is


iter = ProgressBar(1:5)



function fit!(X, y, coeff_i; max_iter = 100 ::Int64)
    global coeff_i  #### guess what we doin' here! declaring this guy inside the function scope!
    iter = ProgressBar(1:max_iter)
    for i in iter
        hessian = Zygote.hessian(coeff_i -> loss(forward(X, coeff_i),y), coeff_i)
        gradient = Zygote.gradient(coeff_i -> loss(forward(X, coeff_i),y), coeff_i)[1]
        coeff_i = coeff_i - (hessian^(-1))*gradient
        if det(hessian) < 0
            @info("Hessian has negative determinant the algorithm will not converge!!!")
            break
        end
        sleep(0.005)
        loss_ = loss(forward(X, coeff_i),y)
        set_description(iter, string(@sprintf("Loss: %.4f", loss_)))
        
end
end

fit!(X, y, coeff_i)   ###### YEEEEEEEEEAAHHHHHHHH Science!
