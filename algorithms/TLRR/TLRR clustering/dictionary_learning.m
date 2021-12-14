function [LL,V] = dictionary_learning(X,opts)
if opts.denoising_flag==0
%% directly using raw data as dictionary
    tho=100;% sigma(i)<=tho*sigma(1)
    [ ~,~,U,V,S ] = prox_low_rank(X,tho);
    LL=tprod(U,S);
else
%% use R-TPCA to denoise data first and then use the recovered data as dictionary
    %% raw R-TPCA algorithm
    [L,~,~,~,~,~] = trpca_tnn(X,opts.lambda,opts);
    %% approximate L, since sometimes R-TPCA cannot produce a good dictionary
    [ U,V,S,~] = tSVDs( L );
    LL=tprod(U,S);
end
