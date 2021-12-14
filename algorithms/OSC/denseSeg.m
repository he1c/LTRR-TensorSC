function t = denseSeg(s,width)
    % Condense the clusters into a single vector
    
    [~, idx] = sort(sum(s,1),'descend');
    s = s(:,idx);
    
    for i=1:size(s,2)
        s(:,i) = i*s(:,i);
    end;
    
    s = sum(s,2);
    t = repmat(s,[1,width]);
end