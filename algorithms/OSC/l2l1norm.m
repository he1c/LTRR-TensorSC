function L = l2l1norm(x)
% L2L1 norm

    L = 0;
    for i=1:size(x,2)
        L = L + norm(x(:,i));
    end
end
