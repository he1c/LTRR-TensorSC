function M_all=mt_blk_sparse_decomb(X,s_X,ndata,K,N)

M_all=cell(N,1);

for i=1:1:K

    M=cell(N,1);

    for j=1:1:ndata

        mc_temp=X(:,1);

        for k=1:1:N      

            M{k}=[M{k} reshape(mc_temp(1:s_X(k,1)*s_X(k,2)),s_X(k,1),s_X(k,2))];

            mc_temp(1:s_X(k,1)*s_X(k,2))=[];

        end

        X(:,1)=[];

    end

    for k=1:1:N
        M_all{k}=[M_all{k};M{k}];
    end
      
end
    
end
