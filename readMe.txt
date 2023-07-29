-------------------------------------------------------

Source Code
-------------------------------------------------------


Specification of dependencies：

The code is written in Python 3.8. Based on the specified software version, OpenCV 3.0, scipy 1.5, Keras 2.0, Tensorflow 2.5, scikit-image 0.17, scikit-learn 0.23 libraries are required for the environment setting. All files need to be on the same directory in order for the algorithms to work.

A demo shows the keyword selection and caption selection process along with the user interface.

To run the algorithm, change the path to the current directory in the command window, and run the [main.py] file:

main.py
The main method that implements the proposed algorithm to perform keyword prediction and caption generation.

The main methods call the following functions:

1. utilModel
Includes methods that define the architecture of the network, customized block and layers of the network.

2. config.m
Includes configurations.

3. utilIO.m
Includes utility methods that reading, writing and processing images and text data.

4. utilMisc.m
Includes utility methods for importing and exporting model parameters.

5. captionTrans.m
Includes methods for training caption generation module.

6. keywordTrans.m
Includes methods for training keyword prediction module.