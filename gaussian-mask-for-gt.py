import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('sample-images/test-round-10.jpg')

kernel = np.ones((5,5),np.float32)/25
dst = cv2.filter2D(img,-1,kernel)
dst2 = cv2.filter2D(dst,-1,kernel)

# plt.subplot(121),plt.imshow(dst2),plt.title('Averaging2')
# plt.xticks([]), plt.yticks([])



# provide guassian blur
gaupic = cv2.GaussianBlur(img,(5,5),cv2.BORDER_DEFAULT)
saliency = cv2.saliency.StaticSaliencyFineGrained_create()
(success, saliencyMap) = saliency.computeSaliency(img)
saliencyMap = (saliencyMap * 255).astype("uint8")
cv2.imshow("Image", img)
cv2.imshow("Output", np.invert(saliencyMap))
cv2.waitKey(0)
#
# plt.subplot(121),plt.imshow(img),plt.title('original image')
# plt.xticks([]), plt.yticks([])
#
# plt.subplot(122),plt.imshow(np.invert(saliencyMap)),plt.title('saliency map')
# plt.xticks([]), plt.yticks([])
# plt.show()


