pySaliencyMap
=============

Python implementation for extracting a saliency map [1] from a still image.

Requirements:

    Python (>= 2.7 is preferable)
    numpy  (>= 1.7 is preferable)
    OpenCV (>= 2.4 is preferable)
    matplotlib (if you would like to run main.py)

Usage:

    If you would like to test this package, please try
        python main-test.py
    This provides a simple example how to use the class pySaliencyMap.
    
    generate-gt-saliency-maps.py generates saliency maps for log images. 
    
    These images nees to be rescaled to size of 260 x 260 or less to the filters to work. 

References:

    [1] L. Itti, C. Koch, E. Niebur, A Model of Saliency-Based Visual Attention for Rapid Scene Analysis, IEEE Transactions on Pattern Analysis and Machine Intelligence, Vol. 20, No. 11, pp. 1254-1259, Nov 1998.
