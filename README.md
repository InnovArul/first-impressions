# first-impressions prediction

- the new links for data download (train, validation data): https://chalearnlap.cvc.uab.cat/dataset/24/description/


The repository contains code, documentation of the code used in **ChaLearn First Impressions Analysis challenge (first round)**

Fact sheet : http://chalearnlap.cvc.uab.es/media/results/None/fact-sheet-evolgen.pdf

Presentation URL : https://drive.google.com/open?id=0BzF_0XI4hJA6dXpRUFc4cVk4VGs

Paper URL : https://drive.google.com/file/d/0B4pMIs_1zlP4YnA3WkxhTEdYSnM/view

challenge URL : https://competitions.codalab.org/competitions/9181

### ChaLearn LAP. Apparent Personality Analysis: First Impressions(First round)

used OS: Ubuntu 14.04

#### Prerequisites for execution:


##### To extract audio features:


1. install ffmpeg
https://git.ffmpeg.org/gitweb/ffmpeg.git
(or)
try "sudo apt-get install ffmpeg"

2. python 2.7 (<--tested) (or above) 

additional packages needed:

```
pip install --user eyed3
pip install --user mlpy
pip install --user scikit-learn
pip install --user progressbar
```

3. Software installations for preparing OpenFace executable
Installation: https://github.com/TadasBaltrusaitis/OpenFace/wiki/Unix-Installation

   - make sure that open face is installed in the folder "OpenFace" of same directory of test files
   i.e., open face should be built and the file "FeatureExtraction" should 
   be available in the folder "data/OpenFace/build/bin"

4. Torch7
Installation: http://torch.ch/docs/getting-started.html


>important modules of Torch7:
>nn, cunn, cutorch, nngraph, rnn, image, optim, xlua, io
>CUDA 7.0 or above


### EXECUTION PROCEDURE:

#### Preprocessing steps:

1. To start preprocessing of data, execute the below command
--------------------------
```
python setup.py
```

After preprocessing, there will be 4 new folders created under "data" folder namely,
```
   "data/trainaudiofeat" - training audio features
   "data/validationaudiofeat" - validation audio features
   "data/trainframes" - training video features
   "data/validationframes" - validation video features
```
2. To train the model,
--------------------------
```
--> in commandline, go to "src" folder,
--> execute "th doall.lua"
```
if you use our code, please cite the paper as below:

```
@inproceedings{baltru2016openface,
  title={Bi-modal First Impressions Recognition using Temporally Ordered Deep Audio and Stochastic Visual Features},
  author={Arulkumar Subramaniam, Vismay Patel, Ashish Mishra, Prashanth Balasubramanian, Anurag Mittal},
  booktitle={ European Conference on Computer Vision (ECCV) Workshop - 2016 on Apparent Personality Analysis},
  pages={-},
  year={2016},
  organization={ECCVW-2016}
}
```
