## fcn

* structuring tf project
```
playground/
├── datasets                        # scripts for preparing datasets
│   ├── cityscapes.py
│   ├── dataloader.py               # factory method for loading datasets
│   ├── flyingthings3d.py
│   ├── kitti_semantics.py
├── fcn
│   ├── model.py
│   ├── README.md
│   ├── run.sh                      # run model training with notes
│   ├── settings.py                 # setup dirs, model name etc.
│   ├── test.py
│   ├── train.py
│   └── vgg16_weights.npz
├── Pipfile
├── Pipfile.lock
└── README.md

mnt/
├── datasets                        
│   ├── cityscapes                  # cityscapes data
│   ├── kitti
├── log
│   ├── <model>
│   |   ├── <run>                   # run 
```
 
### References
* https://github.com/shelhamer/fcn.berkeleyvision.org
* https://github.com/fpanjevic/playground/tree/master/DispNet
* https://www.cs.toronto.edu/~frossard/post/vgg16/
* http://deeplearning.net/tutorial/fcn_2D_segm.html
* https://github.com/mrharicot/monodepth
* https://danijar.com/structuring-your-tensorflow-models/

### Datasets
* http://www.cvlibs.net/datasets/kitti/
* https://www.cityscapes-dataset.com/
