using Zygote
using ForwardDiff
using Plots
using LinearAlgebra
using Distributions
using ProgressBars
using Printf
using Random


X = rand(Uniform(-100,100),(10000000,35))
coeff = randn(36,1)

y = X*coeff[1:35,:] .+ coeff[36,:]   ####This is our data set over all we have 100 samples with 5 dimensions

function forward(X, coeff)
    l = ones(size(X)[1])
    X_ = [X l]
    return X_*coeff
end



coeff_i = randn(36,1) #### This dude is our random coefficients, 

loss(y, y_) = (1/length(y))*sum((y-y_).^2) #### this dude is to be differentiated.
loss(forward(X, coeff_i), y)   #### let's what the loss is





function fit!(X, y, coeff_i; epochs = 10 ::Int64, batch_size = 32::Int64)
    global coeff_i  #### guess what we doin' here! declaring this guy inside the function scope!
    iter = ProgressBar(1:batch_size)
    T = [i for i in 1:size(X)[1]]  #### creating a list here by comprehension
    end_ind = size(T)[1]   #### we get the maximal size!
    batches = Int(round(end_ind/batch_size))    
    iter = ProgressBar(1:batches+1)
    for i in 1:epochs
        shuffle!(T) #### shuffle it
        X_t = X[T,:]
        y_t = y[T]
        for k in iter   #### batch
            x_ = X[batch_size*(k-1)+1:min(end_ind, batch_size*k),:]
            y_ = y[batch_size*(k-1)+1:min(end_ind, batch_size*k)]
            
            hessian = Zygote.hessian(coeff_i -> loss(forward(x_, coeff_i),y_), coeff_i)
            diag = size(hessian)
            gradient = Zygote.gradient(coeff_i -> loss(forward(x_, coeff_i),y_), coeff_i)[1]
            coeff_i = coeff_i - (hessian+10*I(diag[1]))^(-1)*gradient   #### here we do a trick! See the notes????? 
            loss_ = loss(forward(X, coeff_i),y)
            set_description(iter, string(@sprintf("Loss: %.8f",loss_)))
        end
    end
end

coeff_i = randn(36,1) #### This dude is our random coefficients, 

fit!(X, y, coeff_i, epochs = 100000, batch_size = 64)   ###### YEEEEEEEEEAAHHHHHHHH Science!
norm(coeff_i-  coeff)
coeff