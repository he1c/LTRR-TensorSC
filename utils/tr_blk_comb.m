function M=tr_blk_comb(X,ndata)

N=length(X);

M=[];

for i=1:1:N
    mc = mat2cell(X{i}, size(X{i},2)/ndata*ones(ndata,1), size(X{i},2)/ndata*ones(ndata,1));
    mc_t=zeros(ndata,ndata,(size(X{i},2)/ndata)^2);
    
    for k=1:1:ndata
        for j=1:1:ndata
            mc_t(k,j,:)=mc{k,j}(:);
        end
    end
    
    M=cat(3,M,mc_t);
    
end

end