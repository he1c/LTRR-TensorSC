function viewFrame( X, frame, h, w )
    imagesc(reshape(X(:,frame), h, w));
    colormap('gray');
end

