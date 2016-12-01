function  im_result = im_segmentation(im)
%Input: im - the segmented image
%Output: im_result: the sementation result of im
%%
load svm_model.mat
im_o = im2double(im(:,:,1));
se = strel('disk',15);
background = imopen(im_o, se);
im_eq = im_o - background;

% figure, imshow(im_eq, []);

se = strel('disk', 5); 
im = imopen(im_eq, se);
im_contrast = imadjust(im);

im_adaptive = adapthisteq(im_contrast);

% figure, imshow(im_adaptive, []);
level = multithresh(im_adaptive);
im_bw = im2bw(im_adaptive,level);
[moment1, moment2] = moment_calculation(im_eq, im_bw);

test_moment1 = moment1(im_bw);
test_moment2 = moment2(im_bw);
test_intensity = im_eq(im_bw);

test_features = [test_moment1, test_moment2, test_intensity];
test_labels = ones(size(test_moment1));

predict_labels = libsvmpredict( test_labels, test_features, model);

im_result = reconstruction_im(im_bw, predict_labels);

end