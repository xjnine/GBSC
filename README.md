# GBSC
The cluster effect and clustering performance of spectral clustering are improved by using the particle center to construct the spectral clustering similarity matrix.
# Files
These program mainly containing:
  - a  synthetic dataset and real dataset folder named "dataset".
  - four python files
# Requirements
## Installation requirements (Python 3.8)
  - Pycharm 
  - Windows operating system
  - scipy==1.8.1 
  - matplotlib ==3.5.2
  - numpy==1.23.1 
  - psutil ==5.9.1 
  - scikit-learn==1.1.1
  - sklearn==0.0  
  - pandas==1.4.3  
  - seaborn==0.11.2   
# Dataset Format
  - The synthetic dataset is not labeled, and the format is csv. You need to call GranularBallSynthetic to generate a granular-ball.
  - The format of the real dataset is mat. You need to call GranularBallUCI to generate a granular-ball.
# Usage
Run GranularBallSyntheticSC.py to obtain the results of the granular-ball based spectral clustering algorithm on the synthetic dataset, and run GranularBallUCISC.py to obtain the results of the granular-ballbased spectral clustering algorithm on the real dataset.
