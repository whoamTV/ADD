# ADD

## Configration
Edit this file to specify the dataset path and database path.

The subdirectory of "datasetPath" should be "ILSVRC-2012" and "CIFAR10"
```
quick_start_example/config.json
```
## Test the defense performance

To test the defenses for their performance in enhancing model robustness, run
```
python quick_start_example/defense_performance_test.py
```

## Test GADD and DADD
Edit these files to specify the defenses used by GADD/DADD, the default is JPEG.
```
canary_lib/canary_inference_method/GADD.py
canary_lib/canary_inference_method/DADD.py
```
To test GADD, run
```
python quick_start_example/GADD_test.py
```
To test DADD, run
```
python quick_start_example/DADD_test.py
```