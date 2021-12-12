function M=mt_blk_comb(X,ndata)

N=length(X);

M=[];

for i=1:1:N
    
    mc = mat2cell(X{i}, size(X{i},1), size(X{i},2)/ndata*ones(ndata,1));
    
    mc_t=zeros(size(X{i},1)*size(X{i},2)/ndata,ndata);
    
    for k=1:1:ndata        
            mc_t(:,k)=mc{k}(:);
    end
    
    M=[M;mc_t];
    
end
    



end
