"""
=================
Template Matching
=================

We use template matching to identify the occurrence of an image patch
(in this case, a sub-image centered on a single coin). Here, we
return a single match (the exact same coin), so the maximum value in the
``match_template`` result corresponds to the coin location. The other coins
look similar, and thus have local maxima; if you expect multiple matches, you
should use a proper peak-finding function.

The ``match_template`` function uses fast, normalized cross-correlation [1]_
to find instances of the template in the image. Note that the peaks in the
output of ``match_template`` correspond to the origin (i.e. top-left corner) of
the template.

.. [1] J. P. Lewis, "Fast Normalized Cross-Correlation", Industrial Light and
       Magic.

"""
# import numpy as np
# import matplotlib.pyplot as plt
#

from skimage import data
from skimage.feature import match_template
# import time
#
#
# image = data.coins()
# coin = image[170:220, 75:130]
#
#
# start_time = time.time()
# result = match_template(image, coin)
# end_time = time.time()
# print("the image size is :", image.shape)
# print("the match cost:", (end_time - start_time)*1000)
# ij = np.unravel_index(np.argmax(result), result.shape)
# x, y = ij[::-1]
#
# fig = plt.figure(figsize=(8, 3))
# ax1 = plt.subplot(1, 3, 1)
# ax2 = plt.subplot(1, 3, 2)
# ax3 = plt.subplot(1, 3, 3, sharex=ax2, sharey=ax2)
#
# ax1.imshow(coin, cmap=plt.cm.gray)
# ax1.set_axis_off()
# ax1.set_title('template')
#
# ax2.imshow(image, cmap=plt.cm.gray)
# ax2.set_axis_off()
# ax2.set_title('image')
# # highlight matched region
# hcoin, wcoin = coin.shape
# rect = plt.Rectangle((x, y), wcoin, hcoin, edgecolor='r', facecolor='none')
# ax2.add_patch(rect)
#
# ax3.imshow(result)
# ax3.set_axis_off()
# ax3.set_title('`match_template`\nresult')
# # highlight matched region
# ax3.autoscale(False)
# ax3.plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)
#
# plt.show()

import cv2
import numpy as np
import time
from matplotlib import pyplot as plt

img_rgb = cv2.imread('D:/pictures/barcode/3.png')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('D:/pictures/barcode/bar.png', 0)
w, h = template.shape[::-1]

start_time = time.time()
res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
end_time = time.time()
print("match cost : ", (end_time - start_time) * 1000)
threshold = 0.8
start_time = time.time()
result = match_template(img_gray, template)
end_time = time.time()
print("the image size is :", img_gray.shape)
print("the match cost:", (end_time - start_time)*1000)

# umpy.where(condition[, x, y])
# Return elements, either from x or y, depending on condition.
# If only condition is given, return condition.nonzero().
loc = np.where(res >= threshold)

for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

cv2.imwrite('res.png', img_rgb)
