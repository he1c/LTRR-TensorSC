function [ E ] = mysolve_l1l2( W, lambda )
%MYSOLVE_L1L2 Summary of this function goes here
%   Detailed explanation goes here

n = size(W,2);
E = W;
for i=1:n
    E(:,i) = mysolve_l2(W(:,i),lambda);
end

end
