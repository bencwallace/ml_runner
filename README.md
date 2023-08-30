Examples:

```
ml_run -cn fit  # fit with default config settings
ml_run -cn fit dataset=cifar100  # fit to different dataset
ml_run -cn fit model=resnet50  # fit different model
ml_run -cn fit train_loader.batch_size=512  # fit with different batch size

ml_run -cn validate  # validate
```
