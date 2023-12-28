function [Y]=D3_to_2D(data,col)

if (nargin<2)
    col=0;
end

[H,W,B]=size(data);
if col==1    
    Y=reshape(data,[H*W,B])';
end

data_trans=zeros(W,H,B);
if col==0
    for i=1:B
        data_trans(:,:,i)=data(:,:,i)';
    end
    Y=reshape(data_trans,[H*W,B])';
end

end


