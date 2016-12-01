function im = reconstruction_im(im_mask,q)
% reconstruction im using label vector q
% the number of variables in q is the same with one values 
% in im_mask
%%

m=size(im_mask,1);
n=size(im_mask,2);
im=zeros(m,n);
ind=0;
for i=1:n
    for j=1:m
        if im_mask(j,i)==1
            ind=ind+1;
            im(j,i)=q(ind);
        end
    end
end
            