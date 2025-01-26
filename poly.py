__author__ = 'Jervis Muindi'

import random
import cmath
import math

VERBOSE_DEBUG = False

class Poly:
    """
        Represents a polynomial
        With non-negative powers of x
    """
    def __init__(self, pow=0, coeff=[0]):
        """
            Creates a new polynomial
            pow - the highest degree of the polynomial
            coeff - is a listing of (complex) coefficients.
            The first element of this list represents the coefficient of highest x-term in the polynomial
            The last element of this list should represent the constant term.
            Note that all the coefficients should be in the list - if they do have a value, they should be set to 0.
        """
        if (pow != len(coeff) - 1):
            raise ValueError('There is mismatch between power of polynomial and coeffiecnts: %s vs %s' % (pow, coeff))

        self.coeff = []
        # Make a Copy of Values
        for x in coeff:
            self.coeff.append(x)


    def eval(self, x):
        """
            Evaluates the current polynomial at the given point
            x - the point to evaluate at

        """
        result = 0
        curr_pow = self.highest_degree()
        for c in self.coeff:
            result += c * (x ** curr_pow)
            curr_pow -= 1
        return result



    def get_copy(self):
        """
            Returns a copy of this polynomial
        """
        result = []
        for x in self.coeff:
            result.append(x)
        return Poly(self.highest_degree(), result)

    def highest_degree(self):
        """
            Returns the value of the degree of the term with the highest power in the polynomial
        """
        return len(self.coeff) - 1

    def size(self):
        """
            Returns the total number of terms in the polynomial (including the constant and zero-coefficient terms)
        """
        return len(self.coeff)


    def __eq__(self, other):
        """
            Returns true only if all the coefficient exactly match.
        """
        if  (self.size() != other.size()):
            return False
        else:
            size = self.size()
            for i in range(size):
                if (self.coeff[i] != other.coeff[i]):
                    return False
            return True

    def pretty_string(self):
        """
            Produces a pretty representation of the polynomial
        """
        s = ''
        pow = self.highest_degree()
        for x in self.coeff:
            s += '(%s+(%s*I))*x^%s' % (x.real, x.imag, pow)
            pow -= 1
            if pow > -1:
                s += ' + '
        return s

    def __str__(self):
        """
            Produces a string represenation of the polynomial given in terms of a tuple (x,y)
            where x is the coefficient of a term, and y is the power/degree of that term.
        """
        l = list()
        pow = self.highest_degree()
        for x in self.coeff:
            l.append('(%s)*(x^%d)' % (x, pow))
            pow -= 1
        return ' + '.join(l)

    def __repr__(self):
        return self.__str__()

    def __add__(self, other):
        """
            Add this  polynomial to the other one non-destructively.
        """
        big_poly = None
        small_poly = None
        equal_size_poly = False
        if (self.size() > other.size()):
            big_poly = self
            small_poly = other
        elif (self.size() < other.size()):
            big_poly = other
            small_poly = self
        else: # they are equal in size
            equal_size_poly = True

        if (equal_size_poly):
            result = []
            size = self.size()
            for i in range(size):
                val = self.coeff[i] + other.coeff[i]
                result.append(val)
            return Poly(self.highest_degree(), result)
        else:
            result = []
            big_poly_size = big_poly.size()
            small_poly_size = small_poly.size()
            # A sample addition looks like this:
            # [10][20][30][40] - big poly
            #         [55][66] - small poly.
            for i in range(big_poly_size):
                big_poly_reverse_idx = (big_poly_size - 1) - i
                small_poly_reverse_idx = (small_poly_size -1) - i
                if (small_poly_reverse_idx < 0):
                    val = big_poly.coeff[big_poly_reverse_idx]
                    result.append(val)
                else:
                    val = big_poly.coeff[big_poly_reverse_idx] + small_poly.coeff[small_poly_reverse_idx]
                    result.append(val)
            result.reverse()
            return Poly(big_poly.highest_degree(), result)
    

    def get_derivative(self):
        """
            Computes the derivative of this polynomial
        """
        result = []
        size = self.size()

        for i in range(size - 1): # skip constant term
            curr_deg = self.get_power_at_index(i)
            curr_coeff = self.get_x_power_coeff(curr_deg)
            new_coeff = curr_deg * curr_coeff
            result.append(new_coeff)
        new_power = self.highest_degree() - 1 # derivative will be one power lower
        if new_power < 0:
            new_power = 0
        return Poly(new_power, result)


    def normalize(self):
        """
            Divides the polynomial by the leading coefficient
            so that we would have a monic polynomial
        """
        result = []
        norm_const = None

        # Usually the first highest coefficient is good enough. But sometimes that
        # is zero. So we look through all coefficietns to find the first viable one.
        for coeff in self.coeff:
            if coeff != 0:
                norm_const = coeff
                break

        if norm_const is None:
            print("Failed to Normalize Polynomial: %s" % self)

        for coeff in self.coeff:
            val = coeff / (1.0 * norm_const)
            result.append(val)
        return Poly(self.highest_degree(), result)

    def get_highest_degree_of_non_zero_coeff(self):
        """
            Get the degree of the highest term with a non-zero coefficient.
            If all coefficients are zero (polynomial is empty) - then None is returned.
        """
        i = 0
        for coeff in self.coeff:
            if coeff != 0:
                return self.get_power_at_index(i)
            i += 1
        return None

    def strip(self):
       deg  = self.get_highest_degree_of_non_zero_coeff()
       if deg is None:
           return Poly(0, [0.0])
       if deg == self.highest_degree():
           return self
       return Poly(deg, self.coeff[(-deg-1):])

    def get_cauchy_poly(self):
        """
            Returns the Cauchy Polynomial from this polynomial
        """
        first_idx = 0
        last_idx = self.size() - 1
        size = self.size()
        result = []

        do_normalize = False
        norm_const = 0
        for i in range(size):
            if i == first_idx:
                val = self.coeff[i]
                if val != 1:
                    do_normalize = True
                    norm_const = val
                result.append(1)
            elif i == last_idx:
                val = self.coeff[i]
                if do_normalize:
                    val /= 1.0 * norm_const
                val = -abs(val)
                result.append(val)
            else:
                val = self.coeff[i]
                if do_normalize:
                    val /= 1.0 * norm_const
                val = abs(val)
                result.append(val)
        return Poly(self.highest_degree(), result)


    def __mul__(self, other):
        new_degree = self.highest_degree() + other.highest_degree()
        result = [0.0] * (new_degree + 1)
        for i in range(self.highest_degree() + 1):
            for j in range(other.highest_degree() + 1):
                result[i + j] += self.coeff[i] * other.coeff[j]
        return Poly(new_degree, result)
        

    def __divmod__(self, divisor):
        remainder = self.get_copy()
        rem_deg = remainder.highest_degree()
        div_deg = divisor.highest_degree()
        
        result = Poly(0, [0.0])
        while rem_deg >= div_deg:
            curr_deg = rem_deg - div_deg
            poly = Poly(curr_deg, [0.0] * (max(0, curr_deg) + 1))
            quotient = (1.0 * remainder.coeff[-rem_deg-1]) / divisor.coeff[-div_deg-1]
            poly.coeff[-curr_deg-1] = quotient
            remainder -= divisor * poly
            result += poly
            rem_deg -= 1
        
        return result.strip(), remainder.strip()
            
    def __floordiv__(self, divisor):
        return divmod(self, divisor)[0]
    
    def __mod__(self, divisor):
        return divmod(self, divisor)[1]
        

    def divide_linear_poly(self, x_coeff, x_const):
        """
            Divides this polynomial by given linear (1-degree_ polynomial
            x_coeff - coefficient of the (x-)term
            x_const - constant term
        """

        quotient = get_empty_poly(1)
        remainder = self.get_copy()

        num_iterations = remainder.highest_degree()
        dividend_idx = 0
        curr_deg = remainder.highest_degree()
        result = []
        for i in range(num_iterations):
            quotient_coeff = (1.0 * remainder.coeff[dividend_idx]) / x_coeff
            result.append(quotient_coeff)
            term = Term(quotient_coeff,curr_deg - 1)
            poly_term = term.multiply_linear_poly(x_coeff, x_const)
            remainder = remainder.__sub__(poly_term)

            # zero out the highest term just in case we still have residuals
            remainder.set_coeff_at_x_power(curr_deg, 0)

            dividend_idx += 1
            curr_deg -= 1

        return Poly(self.highest_degree() - 1, result)
    
    def __sub__(self, other):
        """
            Does polynomial subtraction in a non-destructive manner.
            Computes this - other
            other - the polynomial to substraction
        """
        neg_poly = other.negate()
        return self.__add__(neg_poly)

    def negate(self):
        """
            Negates this polynomial.
            Does so non-destructively
        """
        result = []
        for x in self.coeff:
            result.append(-x)
        return Poly(self.highest_degree(), result)


    def const_mult(self, c):
        """
            Multiplies through this polynomial by the given constant
            c - constant to multiply
        """
        result = []
        for x in self.coeff:
            val = c * x
            result.append(val)
        return Poly(self.highest_degree(), result)

    def get_power_at_index(self, i):
        """
            Translate the index value to an x-power value
            (i.e. the value of term degree at given position)
            i - the index (0-based)
        """
        max_index = self.size() - 1
        if i < 0 or i > max_index:
            raise ValueError('Invalid index: %s', i)
        max_degree = self.highest_degree()
        return max_degree - i


    def get_x_power_coeff(self, pow):
        """
            Returns the coefficeint of the given x-power
        """
        max_pow = self.highest_degree()
        if pow > max_pow:
            raise ValueError('Invalid Power Arguemnt: %s' % str(pow))
        elif pow < 0:
            raise ValueError('This polynomial does not support negative x-powers')
        else: # it's some other number
            last_idx = self.size() - 1
            pos = last_idx - pow
            return self.coeff[pos]

    def set_coeff_at_x_power(self, pow, val):
        """
            Sets the coefficient of an x-term of given power to the given value
            pow - power of x term
            val - new value of this x-term

            Note that the given power must exist or an error will be thrown
        """
        max_pow = self.highest_degree()
        if pow > max_pow:
            raise ValueError('Invalid Power: %s' % pow)
        elif pow < 0:
            raise ValueError('Negative power arg given: %s' % pow)
        else:
            last_idx = self.size() - 1
            pos = last_idx - pow
            self.coeff[pos] = val
            return self


class Term:
    def __init__(self, coeff=0, deg=0):
        """
            Creates a new term
            deg - degree of the term
            coeff - coefficient of the
        """
        if deg < 0:
            raise ValueError("Degree cannot be negative")
        self.deg = deg
        self.coeff = coeff

    def __str__(self):
        return '(%s,%s)' % (self.coeff, self.pow)
    def __repr__(self):
        return self.__str__()

    def multiply_linear_poly(self, x_coeff, x_const):
        """
            Multiplies this term with the provided linear (1-degree) polynomial
            x_coeff - coefficient of the x^1 term
            x_const - coefficient of the x^0 term

            Returns a Polynomial object
        """
        new_poly_deg = self.deg + 1
        poly = get_empty_poly(new_poly_deg)
        highest_term_coeff = x_coeff * self.coeff
        second_highest_term_coeff = x_const * self.coeff

        poly.set_coeff_at_x_power(new_poly_deg, highest_term_coeff)
        poly.set_coeff_at_x_power(new_poly_deg - 1, second_highest_term_coeff)
        return poly



def get_empty_poly(deg):
    """
        Creates a new empty polynomial of the given degree
    """
    if deg < 0:
        raise ValueError('Invalid polynomial degree')
    size = deg + 1
    result = []
    for _ in range(size):
        result.append(0)
    return Poly(deg, result)

def solve_poly_newton(poly, err):
    """
        Find root of given polynomial by apply newton iteration

        poly - is the polynomial to use
        err - is the maximum error permitted in answer
    """
    x = random.uniform(0,1)
    diff_poly = poly.get_derivative()
    while abs(poly.eval(x)) > abs(err):
        x = x - (poly.eval(x) / (1.0*diff_poly.eval(x)))
    return x


def get_initial_s(poly):
    """
        Computes a random initial s seed to use for beginning of stage 2 of the Jenkins Traub Algorithm
        Seed s is random and multiple calls for a given polynomial will result different complex numbers. Note however,
        per the algorithm, they different complex number would still have the same magnitude.
    """
    cauchy_poly = poly.get_cauchy_poly()
    err = 10 ** (-5)

    beta = solve_poly_newton(cauchy_poly, err)
    rand = random.uniform(0,1) * 2*math.pi
    return abs(beta) * cmath.exp(1j * rand)

def solve_poly_jt(poly, err = 10 **(-5)):
    """
        Finds all the roots (including complex ones) of the given Polynomial by using the Jenkins-Traub
        Algorithm. Roots
        poly - polynomial to solve

        Roots are returned in order of increasing size.
    """

    # A polynomial will have as many roots (including complex ones) as the power
    # of the highest degree term
    num_roots = poly.highest_degree()
    ans = []
    for i in range(num_roots):
        root = solve_smallest_root_poly_jt(poly,err)
        ans.append(root)
        last_run = (num_roots - 1 == i)
        if VERBOSE_DEBUG:
            print('Solving Poly: %s ' % poly)
            print('Found Root: %s' % root)
            print('Negative Root: %s' % (-root))
        if not last_run:
            # Deflate Polynomial to find next largest root
            # on next iteration
            poly = poly.divide_linear_poly(1, -root)
    return ans

def solve_smallest_root_poly_jt(poly, err = 10 ** (-5)):
    """
        Find the smallest of given polynomial by using the Jenkins-Traub Algorithm.
        poly - polynomial to solve.

        Passed in Polynomial is not modified in anyway.
    """
    #TODO(jervis): complete implementing this

    # Ensure that the polynomial is normalized with its leading coefficient equal to 1.
    # (i.e. it is a monic polynomial). Algorithm _won't_ work correctly if this assumption
    # is violated.
    poly = poly.normalize()

    # Stage 1
    # It's good to include this stage in practice though it's not needed theoretically.

    M = 5 # 5 is empirically good for polynomials with degree < 50
    h_poly = poly.get_derivative()
    s = 0
    for i in range(M):
        ev = poly.eval(s)
        if ev == 0:
            break
        # Compute the next H-Polynomial
        const = -h_poly.eval(s) / ev
        pz_poly = poly.const_mult(const)
        adjust_h_poly = h_poly + pz_poly

        # compute the next H-Poly
        h_poly = adjust_h_poly.divide_linear_poly(1, 0)


    # Stage 2
    # ========
    LIMIT = 10 ** 2
    initial_h_poly = h_poly.get_copy()
    t_curr = t_prev = t_next = None
    stage_two_success = False
    root_found = False
    while not root_found: # Retry Loop of whole algorithm
        while not stage_two_success: # Retry Loop for Stage 2
            h_poly = initial_h_poly.get_copy()
            s = get_initial_s(poly) # pick a new s on each retry
            for i in range(LIMIT):
                ev = poly.eval(s)
                if ev == 0:
                    stage_two_success = True
                    break
                # Compute the next H-Polynomial
                const = -h_poly.eval(s) / ev
                pz_poly = poly.const_mult(const)
                adjust_h_poly = h_poly + pz_poly

                # compute the next H-Poly
                next_h_poly = adjust_h_poly.divide_linear_poly(1, -s)

                # Compute the Ts which we use to know when to stop
                h_bar_poly = h_poly.normalize() # normalize polynomial by dividing by leading coefficient
                next_h_bar_poly = next_h_poly.normalize()

                t_curr = s - poly.eval(s) / (1.0 * h_bar_poly.eval(s))
                t_next = s - poly.eval(s) / (1.0 * next_h_bar_poly.eval(s))

                # Termination Test
                if i > 0 and abs(t_curr - t_prev) <= 0.5 * abs(t_prev) and abs(t_next - t_curr) <= 0.5 * abs(t_curr):
                    stage_two_success = True
                    print('Success Stage Two terminated correctly at L = %d' % i)
                    break

                t_prev = t_curr
                h_poly = next_h_poly


            if not stage_two_success:
                print('Failed to terminate correctly in stage 2, retrying ...  ')


        # Stage 3
        # ========

        num_successive_failures = 0
        prev_err = poly.eval(s)
        curr_err = 1

        # Algorithm converges faster than Newton Order 2.
        # So, this limit is way _more_ than enough to find root
        # to the limits of the precision of double (assuming ~10^-300 accuracy)
        # By calculation, starting with an error of 10, we should need
        # only 300 loop iterations to attain this accuracy.
        # If, that is not the case, then stage 3 has failed and we restart from stage two.
        LIMIT =  10 ** 4

        # compute first shifted s
        h_bar_poly = h_poly.normalize()
        s = s - (poly.eval(s) / (1.0 * h_bar_poly.eval(s)))
        prev_s = 0
        stage_3_success = False
        for i in range(LIMIT):
            # Test for convergence / termination
            if abs(poly.eval(s)) < abs(err):
                stage_3_success = True
                break

            # Compute the next H-Polynomial
            const = -h_poly.eval(s) / poly.eval(s)
            pz_poly = poly.const_mult(const)
            adjust_h_poly = h_poly + pz_poly

            # compute the next H-Poly
            next_h_poly = adjust_h_poly.divide_linear_poly(1, -s)
            next_h_bar_poly = next_h_poly.normalize()

            #update the value of s and errors
            prev_s = s
            prev_err = poly.eval(prev_s)

            s = s - (poly.eval(s) / (1.0 * next_h_bar_poly.eval(s)))
            curr_err = poly.eval(s)

            # update h-poly for the next iteration
            h_poly = next_h_poly


            # Test for Errors
            if math.isnan(s.imag) and math.isnan(s.real):
                stage_3_success = False
                break

        if stage_3_success:
            print('Stage 3 was successful')
            print('Root is estimated to be at %s' % s)
            root_found = True
        else: # Restart from Stage 2
            stage_two_success = False
            print(' Stage 3 failure. Restarting algorithm')

    # Return the found root
    return s
