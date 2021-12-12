function M_all=mt_blk_sparse_comb(X,s_X,ndata,K)

N=length(X);

M_all=[];

for i=1:1:N

    M=[];

    for j=1:1:K      

        X_temp=X{i}(1:s_X(i,1),:);
    
        mc = mat2cell(X_temp, size(X_temp,1), s_X(i,2)*ones(ndata,1));
    
        mc_t=zeros(s_X(i,1)*s_X(i,2),ndata);
    
        for k=1:1:ndata        
                mc_t(:,k)=mc{k}(:);
        end
    
        M=[M mc_t];

        X{i}(1:s_X(i,1),:)=[];

    end

    M_all=[M_all;M];
      
end
    
end
