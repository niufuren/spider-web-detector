% perform the cross validation to select the parameters of the svm
% and save the model to svm_model.mat. The training samples are loaded 
% from training_sets.mat
%%
% grid of parameters
load('trainning_sets.mat');
labels_web = ones(length(spiderweb_intensity), 1);
labels_abnormal = ones(length(abnormal_intensity), 1) * -1;
labels = [labels_web; labels_abnormal]

features = [spiderweb_moment1, spiderweb_moment2, spiderweb_intensity; ...
            abnormal_moment1, abnormal_moment2, abnormal_intensity]
folds = 5;
[C,gamma] = meshgrid(0:2:15, 0:2:3);

% grid search, and cross-validation
cv_acc = zeros(numel(C),1);
for i=1:numel(C)
    cv_acc(i) = libsvmtrain(labels, data, ...
                    sprintf('-t 2 -c %f -g %f -v %d', 2^C(i), 2^gamma(i), folds));
end

% pair (C,gamma) with best accuracy
[~,idx] = max(cv_acc);
best_C = 2^C(idx);
best_gamma = 2^gamma(idx);
model = libsvmtrain(labels, data, ...
                 sprintf('-t 2 -c %f -g %f',2^best_C, 2^gamma))
save('svm_model.mat', 'model')