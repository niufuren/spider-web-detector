% im_o = imread('spiderweb2.jpg');
% im = im_o(:,:,1);
% im_bw = imread('spiderweb2_bw.jpg');
% im_bw = im2bw(im_bw(:,:,1));

function [moment1, moment2] = moment_calculation(im, im_bw)
L = bwlabel(im_bw, 8);

nums = max(max(L));

moment1 = zeros(size(im));
moment2 = zeros(size(im));
for i = [1:nums]
    cur_regions = (L ==i);
    eta = SI_Moment(im, cur_regions);
    inv_moments = Hu_Moments(eta);
    moment1 = cur_regions*inv_moments(1) + moment1;
    moment2 = cur_regions*inv_moments(2) + moment2;
%     figure,imshow(cur_regions)
end
% figure,imshow(moment1, []);
% figure,imshow(moment2, []);

end

% figure,imshow(moment1, []);
% figure,imshow(moment2, []);



