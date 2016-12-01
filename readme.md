### Overview
This solution uses SVM to identify the pixels of spiderwebs on frames. The features used for the classifer include: grayscale intensity after the light correction, the first hu's moment and the second hu's moment. The identified spiderwebs are marked in red colour in result folder. When the location of spiderweb is identified, its gray intensity changes between two frames can be ignord. Therefore, the false alert casued by the movement of spiderwebs can be reduced.

The are two versions of algorithm. Malab one was used for a quick prototype; Python code is the proper solution.

### Requirement for Python version:
- install python 2.7
- install libsvm 3.21
- install openCV 3.1

### Run Python code
```python
# get the svm model
python ./python/spiderweb_train.py -i './examples/'  -g './mark/' -o './python/svm_model.out'

#output the segmentation resutl to result.jpg
python ./python/spiderweb_detector.py -i './examples/spiderweb1.jpg' -m './python/svm_model.out' -o './result/result.jpg'
```

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
