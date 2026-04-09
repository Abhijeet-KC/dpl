# Package for Deep Learning projects of MNIST Digit Classifier

### To run the repo as a package
Run in kaggle or colab:
```bash
!pip install git+https://github.com/Abhijeet-KC/dpl
%cd /dpl
!python train.py
```
Run in Windows or linux:
```bash
pip install git+https://github.com/Abhijeet-KC/dpl
cd ./dpl
pip install -e .
python train.py
```
```.

├── README.md
├── LICENSE
├── .gitignore
├── train.py
├── pyproject.toml
├── setup.py
├── src/
│    └── dpl/
│       ├── __init__.py
│       ├── dataset.py
│       ├── model.py
│       └── module.py
├── tests/
│    └──bin/
│       └── module.pydpl
│   ├── __init__.py     
│   └── test_module.py 
└── docs/
     ├── index.html
     └── source.txt
    
```
