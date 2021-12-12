function Y=TRunfold_t(X,k,d)

J=size(X);

N=ndims(X);

order=[k:N 1:k-1];

X_temp=permute(X,order);

Y=reshape(X_temp,prod(J(order(1:d))),[]);

end