A = [i for i in 1:25]
batch_size = 3
end_ind = size(A)[1]

batches = Int(round(end_ind/batch_size))
k = 0
for i in 1:(batches+1)
    len = length(A[batch_size*(i-1)+1:min(end_ind, batch_size*i)])
    println(len)
    
end
println(k)

