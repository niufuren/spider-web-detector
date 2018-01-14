import svmutil as lib
import numpy as np
import spiderweb_detector as sd
import random
import os
import sys
import getopt


class SpiderwebTrain:

    def __init__(self):
        self.web_features = []
        self.non_web_features = []
        self.features = []
        self.labels = []

    def collect_data(self, img_folder, mark_folder):
        names = []
        for file in os.listdir(img_folder):
            if file.endswith(".jpg"):
                names.append(file)

        all_non_web_feature = []
        all_web_feature = []
        for name in names:
            img_name = img_folder + name
            mark_name = mark_folder + name[0:-4] + '_bw_mark.jpg'
            # print mark_name
            spiderweb_detector = sd.SpiderwebDetector()
            spiderweb_detector.read_image(img_name)
            spiderweb_detector.process()
            [web_features, non_web_features] = spiderweb_detector.get_training_features(
                mark_name)

            if len(all_non_web_feature) > 0:
                all_non_web_feature = np.vstack(
                    (all_non_web_feature, non_web_features))
            else:
                all_non_web_feature = non_web_features

            if len(all_web_feature) > 0:
                all_web_feature = np.vstack(
                    (all_web_feature, web_features))
            else:
                all_web_feature = web_features

        self.web_features = all_web_feature.tolist()
        self.non_web_features = all_non_web_feature.tolist()

        web_labels = [1] * len(self.web_features)
        non_web_labels = [-1] * len(self.non_web_features)
        self.labels = web_labels + non_web_labels

        self.features = self.web_features + self.non_web_features
        # np.savetxt('non_web_feature.out', all_non_web_feature)
        # np.savetxt('web_feature.out', all_web_feature)

    def validate_parameters(self, folds=3):

        web_labels = [1] * len(self.web_features)
        non_web_labels = [-1] * len(self.non_web_features)
        labels = web_labels + non_web_labels

        features = self.web_features + self.non_web_features

        C, gamma = np.meshgrid(np.linspace(5, 15, 5), np.linspace(0, 2, 2))

        C = C.flatten()
        gamma = gamma.flatten()

        acc = [0] * np.size(C)

        for i in range(np.size(C)):
            acc[i] = lib.svm_train(
                labels, features,
                "-h 0 -t 2 -c %f -g %f -v %d" % (2**C[i], 2**gamma[i], folds))

        ind = np.argmax(np.asarray(acc))

        return [C[ind], gamma[ind]]

    def train_svm(self, C, gamma, model_output):

        labels = self.labels
        features = self.features

        model = lib.svm_train(
            labels, features, "-h 0 -t 2 -c %f -g %f" % (2**C, 2**gamma))
        lib.svm_save_model(model_output, model)


def main(argv):

    image_folder = ''
    groundtruth_folder = ''
    model_output = ''

    try:
        opts, args = getopt.getopt(argv, "i:g:o:")
    except getopt.GetoptError:
        print 'spiderweb_train.py -i <image_folder> -g <markfile_folder> -o <svm_model.out>'
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-i':
            image_folder = arg
        elif opt == '-g':
            groundtruth_folder = arg
        elif opt == '-o':
            model_output = arg

    spiderweb_train = SpiderwebTrain()

    spiderweb_train.collect_data(image_folder, groundtruth_folder)
    C, gamma = spiderweb_train.validate_parameters()
    spiderweb_train.train_svm(C, gamma, model_output)


if __name__ == "__main__":
    main(sys.argv[1:])
