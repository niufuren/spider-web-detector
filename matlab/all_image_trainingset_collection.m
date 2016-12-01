% select the training samples from the images in the folder of example,
% the ground truth is marked in images in the folder of mark, and
% the collected training sample set is saved in training_sets.mat 
%%

image_folder = 'examples/*.jpg';
folder_name = 'examples/';
folder_mark = 'mark/';
file_list = dir(image_folder);
spiderweb_moment1 = [];
spiderweb_moment2 = [];
spiderweb_intensity = [];

abnormal_moment1 = [];
abnormal_moment2 = [];
abnormal_intensity = [];

for ind = 1:length(file_list)
    name = file_list(ind).name;
    im_name = strcat(folder_name,name);
    im = imread(im_name);
    
    mask_name = strcat(name(1:end-4), '_bw_mask.jpg');
    mask_name = strcat(folder_mark, mask_name);
    mask_img = imread(mask_name);
    [img_spiderweb_moment1, img_spiderweb_moment2, img_spiderweb_intensity,...
     img_abnormal_moment1, img_abnormal_moment2, img_abnormal_intensity]...
     = im_trainningset_collection(im, mask_img);
    
    spiderweb_moment1 = [spiderweb_moment1; img_spiderweb_moment1];
    spiderweb_moment2 = [spiderweb_moment2; img_spiderweb_moment2];
    spiderweb_intensity = [spiderweb_intensity; img_spiderweb_intensity];
    
    abnormal_moment1 = [abnormal_moment1; img_abnormal_moment1];
    abnormal_moment2 = [abnormal_moment2; img_abnormal_moment2];
    abnormal_intensity = [abnormal_intensity; img_abnormal_intensity];
end

save('training_sets.mat', 'spiderweb_moment1', 'spiderweb_moment2',...
    'spiderweb_intensity', 'abnormal_moment1', 'abnormal_intensity'); 
    