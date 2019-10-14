# forward AD
using ForwardDiff
f(x::Vector) = x[1] + 10*x[2].^2
x = [1, 2]
#gradient = ForwardDiff.gradient(f, x)
hessian = ForwardDiff.hessian(f, x)
# reverse AD
using Zygote
m(x::Vector) = x[1] + 10*x[2].^2
gradient2 = Zygote.gradient(m, x)
#hessian2 = Zygote.hessian(m, x)
