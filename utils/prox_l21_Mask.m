function Y=prox_l21_Mask(X,lambda,Mask)

N=size(X,2);

Y=zeros(size(X));

for i=1:1:N
    
    temp=X(Mask(:,i)>0,i);
    nrm=norm(temp);
    
    if nrm>lambda
        Y(:,i)=(nrm-lambda)/nrm*Mask(:,i).*X(:,i);
    end
    
    Y(:,i)=Y(:,i)+(1-Mask(:,i)).*X(:,i);
    
end


end