%perform the segmentaion for all the images in the folder of examples

image_folder = 'examples/*.jpg';
folder_name = 'examples/';
file_list = dir(folder);

for ind = 1:length(file_list)
    name = strcat(folder_name, file_list(ind).name);
    im = imread(name);
    im_result = im_segmentation(im);
    figure,imshow(im_result);
end
