function M=mt_blk_decomb(X,N,ss)

ndata=size(X,2);

M=cell(1,N);

for i=1:1:N
    
    s_mt=ss(i,1)*ss(i,2);
    temp=[];
    
    for j=1:1:ndata
         temp=[temp reshape(X(1:s_mt,j),ss(i,1),ss(i,2))];
    end
    
    M{i}=temp;
    
    X(1:s_mt,:)=[];
    
end

end