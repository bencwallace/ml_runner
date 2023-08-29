Examples:

```
ml_fit  # fit with default config settings
ml_fit train_loader.batch_size=128  # fit with different batch size
ml_fit dataset._target_=torchvision.datasets.CIFAR100  # fit to different dataset
ml_fit pl_module.model._target_=torchvision.models.resnet18  # fit different model
```