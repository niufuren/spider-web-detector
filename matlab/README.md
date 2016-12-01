### Overview

This is a quick prototype code to test the algorithm. Proper solution can be found at https://github.com/niufuren/spider-web-detector

### Requirement for Matlab version:
- install libsvm 3.21

### Run Matlab code
```
%collect the trainning samples
all_image_trainingset

%train the svm model
svm_parameters_selection

%perform the segmentation for all the images in example folder, and show the result
all_image_segmentation
```
