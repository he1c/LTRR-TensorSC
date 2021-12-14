function y = normr(x)
%BASED ON MATLAB CODE

    xi = x;
    xi(~isfinite(xi)) = 0;
    len = sqrt(sum(xi.^2,2));
    yi = bsxfun(@rdivide,xi,len);
    zeroRows = find(len==0);
    if ~isempty(zeroRows)
        numColumns = size(xi,2);
        row = ones(1,numColumns) ./ sqrt(numColumns);
        yi(zeroRows,:) = repmat(row,numel(zeroRows),1);
    end
    y = yi;

end
