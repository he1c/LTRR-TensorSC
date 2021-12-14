function G = clusterUknown(XO,U)
% Clusters the partially observed data points in XO according to the
% subspaces in U.


K = length(U);
[~,N] = size(XO);

residuals = zeros(K,N);
for i=1:N,
    oi = XO(:,i)~=0;
    for k=1:K,
        xoi = XO(oi,i);
        Uoi = U{k}(oi,:);
        residuals(k,i) = norm(xoi-Uoi/(Uoi'*Uoi)*Uoi'*xoi,'fro')/norm(xoi,'fro');
    end
end

[~,G] = min(residuals);
G = G';

end