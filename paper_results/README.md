## Reproduction of results from paper

### Installation
```
git clone https://github.com/vlad-user/parallel-tempering
```

To reproduce the result just run each individual file. The resulted plots will be generated in subdirectory `plots`.

## Examples:

### Lenet5 | varying dropout vs fixed dropout | CIFAR-10 dataset
```
python lenet5_cifar10_dropout.py
```
![](_images/cifar-dropout.png=80x80)

### Lenet5 | varying dropout vs fixed dropout | EMNIST-letters dataset
```
python lenet5_emnist_dropout.py
```
![](_images/emnist-dropout.png=80x80)
### Lenet5 | varying learning rate vs fixed learning rate | CIFAR-10 dataset
```
python lenet5_cifar10_learning_rate.py
```
![](_images/cifar-lr.png=80x80)
### Lenet5 | varying learning rate vs fixed learning rate | EMNIST-letters dataset
```
python lenet5_emnist_learning_rate.py
```
![](_images/emnist-lr.png=80x80)