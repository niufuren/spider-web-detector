import cv2
import numpy as np
import svmutil as lib
import sys
import getopt


class SpiderwebDetector:

    def __init__(self):
        self.img = []
        self.im_bw = []
        self.img_eq = []
        self.moment1_mat = []
        self.moment2_mat = []
        self.labels = []

    def read_image(self, file_name):

        img = cv2.imread(file_name)
        [_, _, r] = cv2.split(img)
        self.img = r

    def imadjust(self, img):

        sorted_img = np.sort(img, axis=None)
        number_ele = np.size(sorted_img)

        low_in = sorted_img[int(number_ele * 0.02)]
        high_in = sorted_img[int(number_ele * 0.98)]

        low_out = 0
        high_out = 255

        new_img = low_out + (high_out - low_out) * \
            np.true_divide((img - low_in), (high_in - low_in))

        for i in range(0, len(new_img)):
            for j in range(0, len(new_img[0])):
                if new_img[i][j] >= 255:
                    new_img[i][j] = 255

        new_img = np.array(new_img, dtype=np.uint8)

        return new_img

    def get_pixels_in_contour(self, contour):
        cimg = np.zeros_like(self.img)
        pixels_in_contour = []
        cv2.drawContours(cimg, [contour], 0, color=255,
                         thickness=-1, offset=(0, 0))
        cv2.drawContours(cimg, [contour], 0, color=255,
                         thickness=2, offset=(0, 0))
        pixels = np.where(cimg == 255)
        pixels_in_contour.append(pixels)
        return pixels_in_contour

    def regions_moment(self):
        im_bw = self.im_bw.copy()
        im2, contours, hierarchy = cv2.findContours(im_bw,
                                                    cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE,
                                                    offset=(0, 0))

        moments_list = [None] * len(contours)
        for i in range(0, len(contours)):
             # print contours[i]
            moments_feature = [None] * 3
            pixels = self.get_pixels_in_contour(contours[i])

            moments_feature[0] = pixels
            moments = cv2.moments(contours[i])
            hu_moments = cv2.HuMoments(moments)
            moments_feature[1] = hu_moments[0]
            moments_feature[2] = hu_moments[1]
            moments_list[i] = moments_feature

        return moments_list

    def process(self):
        kernel_tophat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (29, 29))
        img_eq = cv2.morphologyEx(self.img, cv2.MORPH_TOPHAT, kernel_tophat)
        self.img_eq = img_eq.copy()
        # cv2.imwrite('img_eq.jpg', img_eq)

        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

        img_open = cv2.morphologyEx(img_eq, cv2.MORPH_OPEN, kernel_open)

        # perform the imadjust
        img_contrast = self.imadjust(img_open)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_adaptive_contrast = clahe.apply(img_contrast)

        _, im_bw = cv2.threshold(img_adaptive_contrast, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        self.im_bw = im_bw.copy()

        moments_list = self.regions_moment()

        self.moment1_mat, self.moment2_mat = self.construct_moment_mat(
            moments_list)

    def spiderweb_identify(self, model_name):

        model = lib.svm_load_model(model_name)
        regions = np.where(self.im_bw == 255)

        moment1_feature = self.moment1_mat[regions]

        moment2_feature = self.moment2_mat[regions]

        intensity_feature = self.img_eq[regions]
        # intensity_features = intensity_feature/255.0

        features = np.vstack((moment1_feature, moment2_feature,
                              intensity_feature))
        features = features.T
        features = features.tolist()

        labels = [1] * len(features)

        p_label, _, _ = lib.svm_predict(labels, features, model, '-q')
        self.labels = p_label

    def save_result(self, output_image):
        im_rgb = cv2.cvtColor(self.img, cv2.COLOR_GRAY2RGB)

        regions = np.where(self.im_bw == 255)

        for i in range(len(regions[0])):
            x = regions[0][i]
            y = regions[1][i]

            if self.labels[i] > 0:
                im_rgb[:, :, 2][x, y] = 255
                im_rgb[:, :, 1][x, y] = 0
                im_rgb[:, :, 0][x, y] = 0

        cv2.imwrite(output_image, im_rgb)

    def construct_moment_mat(self, moments_list):
        moment1_mat = np.zeros(np.shape(self.img))
        moment2_mat = np.zeros(np.shape(self.img))

        for i in range(len(moments_list)):
            region = moments_list[i][0]
            # print region
            moment1_mat[region[0][0], region[0][1]] = moments_list[i][1]
            moment2_mat[region[0][0], region[0][1]] = moments_list[i][2]

        return [moment1_mat, moment2_mat]

    def get_training_features(self, mask_img):
        mark_img = cv2.imread(mask_img)
        # b, g, r = cv2.split(mark_img)

        r = mark_img[:, :, 2]
        g = mark_img[:, :, 1]

        # get the non_web marker
        _, non_web = cv2.threshold(r, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        reverse_non_web = 255 - non_web

        # get the web marker
        _, web = cv2.threshold(g, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        reverse_web = 255 - web

        non_web_bw = np.bitwise_and(np.equal(self.im_bw, 255),
                                    np.equal(non_web, 255))

        non_web_bw = np.bitwise_and(non_web_bw,
                                    np.equal(reverse_web, 255))
        # cv2.imwrite('non_web' + str(self.count) + '.jpg', non_web_bw * 255)

        web_bw = np.bitwise_and(np.equal(self.im_bw, 255),
                                np.equal(web, 255))
        web_bw = np.bitwise_and(web_bw,
                                np.equal(reverse_non_web, 255))

        # get the non_web_feature coordinates
        non_web_pixels = np.where(non_web_bw == 1)

        moment1_feature = self.moment1_mat[non_web_pixels]

        moment2_feature = self.moment2_mat[non_web_pixels]

        intensity_feature = self.img_eq[non_web_pixels]
        # intensity_feature = intensity_feature / 255.0

        non_web_features = np.vstack((moment1_feature, moment2_feature,
                                      intensity_feature))
        non_web_features = non_web_features.T

        # get the web feature
        web_pixels = np.where(web_bw == 1)

        moment1_feature = self.moment1_mat[web_pixels]

        moment2_feature = self.moment2_mat[web_pixels]

        intensity_feature = self.img_eq[web_pixels]

        web_features = np.vstack((moment1_feature, moment2_feature,
                                  intensity_feature))

        # print web_features.shape
        web_features = web_features.T
        # print web_features.shape

        return [web_features, non_web_features]


def main(argv):
    input_image = ''
    model = ''
    output_image = ''

    try:
        opts, args = getopt.getopt(argv, 'i:m:o:')
    except getopt.GetoptError:
        print 'spiderweb_detector.py -i <input_image> -m <svm_model> -o <output_file>'
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-i':
            input_image = arg
        elif opt == '-m':
            model = arg
        elif opt == '-o':
            output_image = arg

    spiderweb_detector = SpiderwebDetector()
    spiderweb_detector.read_image(input_image)
    spiderweb_detector.process()

    spiderweb_detector.spiderweb_identify(model)
    spiderweb_detector.save_result(output_image)


if __name__ == "__main__":
    main(sys.argv[1:])
