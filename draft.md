#Draft for Master Thesis
## What's Automatic differentiation
* It's used to get the gradients (or derivatives) of the functions. We did a lot of symbolic differentiations in the past, like we also know if the equation is y=x^2 then the derivative is y'=2*x. This is symbolic differentiation. Since it's not always continuous especially in the computer, we use something to approximate the derivatives, for instance is we are going to calculate the same derivative. The it would be y' = (y(x1)-y(x2))/delta(x), given x is small enough. The method can be accurate enough if the steps are small enough. However, if the steps are too small, it might cause some numerical errors given the delta(x) as the denominator is close to zero and the derivative can be really large if it's beyond the ability of the calculator. (round of error, float point error, etc...)

And here comes to the reason why we use automatic differentiation (AD), cause it's kind of partially symbolic and partially numerical. Compared with SD, we can calculate the derivative very quickly and without using up the memory, and compared with ND, we can get relatively better quality of the derivatives.

In principle, AD performs a non-standard interpretation of a given computer program by replacing the domain of the variables to incorporate derivative values and redefining the semantics of the operators to propagate derivatives via the chain rule of differential calculus. Despite its widespread use in other fields, general-purpose AD has been underused by the machine learning community until very recently. (Back-propagation in the ANN is one form of the reverse mode of AD). And it is also because of the state of the art of the community, a lot of AD packages are now available, however this method has not been widely used in the environmental models. This is the original motivation of the thesis.

* It's a transformation of algorithms for functions into functions for their derivatives;

* Apply symbolic differentiation at the elementary level, and keep intermediate numerical results, in lock step with the evaluation of the main function.


## Environmental models available
1, Linear regression model
2, Batch reactor model
3, The nitrification model from the paper from Chris
4, Conceptual model + the MOD2 UMWx (Citydrain2)

## Performance metric (suggestions)
1, Accuracy
2, Run time and memory
3, Applicability
4, Ease of implementation and readability

## why not choosing continuous sensitive analysis (CSA)
No additional code was required to be written by the AD method. Whereas for the CSA, whenever the user is changing the number of ODEs during an event, and events can change solver internals like the current integration time. CSA would require a separate implementation for each of these equations which could be costly to developer time.

Adjoint sensitivity analysis is used to find the gradient of the solution with respect to some functional of the solution.

CAS which saves a continous solution for the forward pass of the solution and utilizes its interpolant in order to calculate the requisite Jacobian and gradients for the backwards pass.

It is equivalent to "backpropogation" or reverse-mode automatic differentiation of a differential equation.
