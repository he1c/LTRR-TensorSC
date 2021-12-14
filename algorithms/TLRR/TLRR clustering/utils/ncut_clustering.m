function [mean_nmi,grps] = ncut_clustering(Z, labels,t )
warning off;

K = length(unique(labels));

[n1,n2,n3]=size(Z);
Z=abs(Z);
Z=(Z+tran(Z))./2.0;

XX=zeros(n1,n2);
for i=1:n3
    XX=XX+0.5*(Z(:,:,i)+Z(:,:,i)');
end



[~,S,U] = svd(XX,'econ');
S = diag(S);
r = sum(S>1e-3*S(1));
U = U(:,1:r);S = S(1:r);
U = U*diag(sqrt(S));
U = normr(U);
L = (U*U').^t;
% spectral clustering
D = diag(1./sqrt(sum(L,2)));
L = D*L*D;


d=10;
results=zeros(d,3);
for p=1:d
    grps = SpectralClustering(L,K);
    [result] = ClusteringMeasure(labels,grps);
    results(p,:)=result;
end
mean_nmi=mean(results);
end

