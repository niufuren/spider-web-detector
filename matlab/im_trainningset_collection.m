
function [spiderweb_moment1, spiderweb_moment2, spiderweb_intensity...
    abnormal_moment1, abnormal_moment2,abnormal_intensity] = im_trainningset_collection(im,mask_img)
%collect the training sets marked by mask_img
%Input: im - the original image
%       mask_img - the image marked the ground truth
%Output:spiderweb_moment1: the first Hu's invariant moment of spiderweb
%       spiderweb_moment2: the second Hu's invariant moment of spiderweb
%       spiderweb_intensity: the grayscale of spiderweb after light
%       correction
%       abnormal_moment1: the first Hu's invariant moment of non-spiderweb 
%       abnormal_moment2: the second Hu's invarinat moment of non-spiderweb
%       abnormal_intensity: the grayscale of non-spiderweb after light
%       correction
%%
im_o = im2double(im(:,:,1));
se = strel('disk',15);
background = imopen(im_o, se);
im_eq = im_o - background;

% figure, imshow(im_eq, []);

se = strel('disk', 5); 
im = imopen(im_eq, se);
% figure, imshow(im, []);
im_contrast = imadjust(im);

im_adaptive = adapthisteq(im_contrast);


level = multithresh(im_adaptive);
im_bw = im2bw(im_adaptive,level);

[moment1, moment2] = moment_calculation(im_eq, im_bw);

%get the training mask

abnormal_r = im2bw(mask_img(:,:,1));
abnormal_g = im2bw(mask_img(:,:,2));
abnormal_mask = abnormal_r & (~abnormal_g) & im_bw;
%abnormal set
abnormal_moment1 = moment1(abnormal_mask);
abnormal_moment2 = moment2(abnormal_mask);
abnormal_intensity = im_eq(abnormal_mask);

%spiderweb set
spiderweb_mask = abnormal_g &(~abnormal_r)& im_bw;

spiderweb_moment1 = moment1(spiderweb_mask);
spiderweb_moment2 = moment2(spiderweb_mask);
spiderweb_intensity = im_eq(spiderweb_mask);

end

