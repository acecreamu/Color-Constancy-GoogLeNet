# Artificial Color Constancy via GoogLeNet with Angular Loss Function

Supporting code to the paper<br>
[O Sidorov. Artificial Color Constancy via GoogLeNet with Angular Loss Function](https://arxiv.org/abs/1811.08456)

![image preview](https://github.com/acecreamu/color-constancy-googlenet/blob/master/img.jpg)

### Requirements
The code is designed for MATLAB R2017b with Neural Networks Toolbox. It should also work with other recent versions.

Running the code requires having installed pre-trained GoogLeNet model. (Can be installed within MatLab using Add-On explorer, or externally from [File Exchange](https://www.mathworks.com/matlabcentral/fileexchange/64456-deep-learning-toolbox-model-for-googlenet-network))

### Structure
Most of the operations required for modification the network, loading the data, and performing  a training, is contained in `main.m`.

`angularRegressionLayerL#.m` contains description of regression layers with custom loss function. Which one to use can be specified in main.m, line 7 (L2 by default).
