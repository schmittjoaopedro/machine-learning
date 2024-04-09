from numpy.linalg import linalg


# Steepest Descent
# We can do better by moving in a direction proportional to the Jacobian, down the slope.
# We can set an aggression parameter, γ, for how big the jumps should be. i.e.  δx=−γJ
def next_step(f, Jacobian, Hessian):
    gamma = 0.5
    return -gamma * Jacobian


# Hessian
# The trouble with the previous method is it is not always clear how big to set γ to be.
# Too big, and the jumps are too big, missing all the features. Too small, and it will
# take too long to converge.
#
# A way of automatically determining the jump size, is to use the Hessian, i.e., the second
# derivative matrix. Then, the step size can be given as,  δx=−H^−1 * J
#
# This not only sets the step size, but can also change the direction too. Be careful,
# this method is just as likely to find maxima as it is minima.
def next_step(f, Jacobian, Hessian):
    return -linalg.inv(Hessian) @ Jacobian

# Hybrid method
#
# You may have noticed, that if you are sufficiently close to a stationaty point already,
# the Hessian method will find it in relatively few steps. Though in most cases, the step
# size is too large, and can even change the direction up hill.
#
# We can try a hybrid method which tries the Hessian unless the step would be too big, or
# it would point backwards, in which case it goes back to using steepest descent.
#
# See if you think this is any better.
def next_step(f, J, H) :
    gamma = 0.5
    step = -linalg.inv(H) @ J
    if step @ -J <= 0 or linalg.norm(step) > 2 :
        step = -gamma * J
    return step
