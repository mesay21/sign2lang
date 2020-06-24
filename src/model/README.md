# Contents of the directory
model.py: module that contains Model class.  The class loads the I3D RGB model without the top layer, adds a custom
classifier, and returns a keras model.

i3d_inception.py: module that defines the I3D model.  Modified the source code from (https://github.com/dlpbc/keras-kinetics-i3d) to work tensorflow-2.2.0.

