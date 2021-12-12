function Y=TRfold_t(X,k,J,N)

order=[k:N 1:k-1];
X_temp=reshape(X,J(order));
Y=ipermute(X_temp,order);

end