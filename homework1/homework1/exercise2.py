
def gradient_step(xk, df, sigma):
    """Return the next interate xk+1 given the previous
    xk, the derivate function f' and the sigma which is
    a scaling factor.

    Parameters
    _________
    xk : float
    df : functions
    sigma : float

    Return
    _________
    xk1 : float
    """
    xk1 =xk -sigma*df(xk)
    return xk1

def gradient_descent(f, df, x, sigma=0.5, epsilon=1e-8):
    """Returns a minima of `f` using the Gradient Descent method.

    A local minima, x*, is such that `f(x*) <= f(x)` for all `x` near `x*`.
    This function returns a local minima which is accurate to within `epsilon`.

    `gradient_descent` raises a ValueError if `sigma` is not strictly between
    zero and one.

    Parameters
    _________
    f : function
    x0 : float
    df : function
    sigma : float
    epsilon :float
    Return
    _________
    xk1 : float
    The minima of `f`
    """
    if sigma <=0 or sigma >=1:
        raise ValueError('')

    xk1 = x
    xk = x+1
    while abs(xk1-xk) > epsilon:
        xk = xk1
        xk1= gradient_step(xk,df,sigma)

    if f(xk1) <= f(x):
        return xk1
    else: #the case when xk1 it's local maximum
        return 99