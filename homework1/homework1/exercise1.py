# the documenation has been written for you in this exercise

def collatz_step(n):
    """Returns the result of the Collatz function.

    The Collatz function C : N -> N is used in `collatz` to generate collatz
    sequences. Raises an error if n < 1.

    Parameters
    ----------
    n : int

    Returns
    -------
    int
        The result of C(n).

    """
    if n < 1:
        raise ValueError("number is less than one")
    if n % 2 == 0:
        n = n/2
        return n

    elif n % 2 == 1:
         if n ==1:
            return n

         n = 3*n+1
         return n

def collatz(n):
    """Returns the Collatz sequence beginning with `n`.

    It is conjectured that Collatz sequences all end with `1`. Calls
    `collatz_step` at each iteration.

    Parameters
    ----------
    n : int

    Returns
    -------
    sequence : list
        A Collatz sequence.

    """
    c=[k]

    while k >1:
         c = c+ [collatz_step(k)]
         k = collatz_step(k)

    return c
