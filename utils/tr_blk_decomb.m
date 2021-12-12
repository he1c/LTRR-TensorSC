function M=tr_blk_decomb(X,N,ss)

ndata=size(X,1);

M=cell(1,N);

for i=1:1:N
    
    temp=cell(ndata,ndata);
    s_t=ss(i,2)^2;
    
    
    for j=1:1:ndata
        for k=1:1:ndata
            temp{j,k}=reshape(X(j,k,1:s_t),ss(i,2),ss(i,2));
        end
    end

    X(:,:,1:s_t)=[];
    
    M{i}=cell2mat(temp);
    
end

end