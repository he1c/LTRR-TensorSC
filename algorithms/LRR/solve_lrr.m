function [Z,E] = solve_lrr(X,lambda)
Q = orth(X');
A = X*Q;
[Z,E] = lrra(X,A,lambda,0);
Z = Q*Z;