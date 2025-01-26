import argparse
from poly import Poly, solve_poly_jt
from numpy import roots, argmin


__author__ = 'Jervis Muindi'
# Date: May 2013
# Numerical Analysis and Algorithms
# Extra Credit
# Jenkins Traub Algorithm


def signum(x):
    if isinstance(x, complex):
        if x == x.real:
            x = x.real
    return 1 if x > 0 else (-1 if x < 0 else 0)

def sturm_chain_num_real_roots(p, a, b):
    # https://math.stackexchange.com/a/4087390/1230506
    # number of real roots in half-open interval [a, b)
    chain = [p, p.get_derivative()]
    while True:
        i_prev = len(chain)-1
        p_new = (chain[i_prev-1] % chain[i_prev]).const_mult(-1)
        chain.append(p_new)
        if p_new.get_highest_degree_of_non_zero_coeff() is None:
            break
    
    sigma_a = 0
    sigma_b = 0
    prev_sign_a = None
    prev_sign_b = None
    for p in chain:
        val_a = p.eval(a)
        val_b = p.eval(b)
        
        sign_a = signum(val_a)
        sign_b = signum(val_b)
        
        if prev_sign_a is not None and sign_a != prev_sign_a:
            sigma_a += 1
            
        if prev_sign_b is not None and sign_b != prev_sign_b:
            sigma_b += 1
        
        prev_sign_a = sign_a
        prev_sign_b = sign_b
    
    return abs(sigma_a - sigma_b)


def cauchy_bound(p):
    # https://math.libretexts.org/Bookshelves/Precalculus/Precalculus_(Stitz-Zeager)/03%3A_Polynomial_Functions/3.03%3A_Real_Zeros_of_Polynomials
    M = 0
    for c in p.coeff:
        m = abs(c) / abs(p.coeff[0])
        if m > M:
            M = m
    return (-M-1, M+1)

    
def newton_raphson_method(p, x0, xmax, eps=1e-15):
    p_deriv = p.get_derivative()
    x = x0
    err = p.eval(x).real
    failed = False
    while abs(err) > eps:
        err = p.eval(x).real
        y = p_deriv.eval(x).real
        if y == 0:
            x -= err
        else:
            x -= max(-0.5*xmax, err / y)
        #print(x, err, y)
        if x >= xmax:
            print("Failed to find smallest positive real root using Newton method!")
            failed = True
            break
    if x <= 0:
        return newton_raphson_method(p // Poly(1, [1, -x]), x0, xmax)
    return x if not failed else None
    

def main():

    poly_help_msg = 'List of Coefficients of the polynomial to find the roots for. Start from the highest power and proceed in a descending order until the constant term. All coefficients must be specified and not skipped. The symbol \'j\' can be used to denote a complex number coefficient. Example:1+2j. Number coefficient must be separated by space. '

    parser = argparse.ArgumentParser(description='General Polynomial Root Solver. It applies the Jenkins-Traub Algorithm')
    parser.add_argument('-p', '--polynomial', nargs='+', type=complex, required=True,
                        help=poly_help_msg)
    parser.add_argument('-e', '--error', type=float)

    args = vars(parser.parse_args())
    poly_coeff = args['polynomial']

    err = 10 ** (-15) # Default Error Values
    if args['error']:
        err=args['error']


    poly_pow = len(poly_coeff) - 1
    poly = Poly(poly_pow, poly_coeff)

    print('Finding Roots for the Polynomial:\n %s\n' % poly.pretty_string())
    print('Using Error Value of: %s\n' % err)
    
    bound = cauchy_bound(poly)
    print('Cauchy\'s bound on real roots:', bound)
    
    num_real_roots = sturm_chain_num_real_roots(poly, *bound)
    print('Sturm\'s chain - number of real roots:', num_real_roots)
    if num_real_roots > 0:
        smallest_pos_real_root = newton_raphson_method(poly, 0, bound[1])
        print('Newton-Raphson - smallest positive real root:', smallest_pos_real_root)
    print()

    print('Starting Root Search ...')
    ans = solve_poly_jt(poly)
    print('Root Search Complete')

    print('\n*********************\n')
    print('For The Polynomial\n%s\nThe roots found in order of increasing magnitude are:' % poly.pretty_string())
    counter = 1
    for root in ans:
        root_real = round(root.real*1e3)/1e3
        root_imag = round(root.imag*1e3)/1e3
        root_type = "complex" if root_real != 0 and root_imag != 0 else ("real" if root_imag == 0 else "imag")
        print ('%d) %s (%s root)' % (counter, root, root_type))
        counter += 1
        
    print('\n*********************\n')
    print('Comparison with numpy.roots - abs errors:')
    roots_np = roots(poly_coeff)
    counter = 1
    for root in ans:
        root_np = roots_np[argmin([abs(root-root_np) for root_np in roots_np])]
        print ('%d) %s (error: %s)' % (counter, root_np, abs(root-root_np)))
        counter += 1

if __name__ == '__main__':
    main()