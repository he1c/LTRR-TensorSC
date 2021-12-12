function ind=tcu_rearrange(n_t,k,d)

    N=length(n_t);
    order=[k:N 1:k-1];
    ind_temp=[];
    for c=1:1:n_t(end)
       if N-k+1>d
            temp1=prod(n_t(order(d+1:N-k)));
            temp2=prod(n_t(order(N-k+2:N)));
       else
            temp1=prod(n_t(order(1:N-k)));
            temp2=prod(n_t(order(N-k+2:d)));
       end
       for p=1:1:temp2
            ind_temp=[ind_temp temp1*((p-1)*n_t(end)+c-1)+1: temp1*((p-1)*n_t(end)+c)];
       end
    end
    ind=ind_temp;
    
end