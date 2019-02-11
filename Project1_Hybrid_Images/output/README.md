# Project 1

The following JSON object is the config used by the program to generate the output hybrid image:

```json
{
  "right_size": 3, 
  "left_sigma": 2.0, 
  "right_sigma": 5.3, 
  "right_mode": "high", 
  "view_grayscale": 0, 
  "left_mode": "low", 
  "left_size": 7, 
  "mixin_ratio": 0.55, 
  "save_grayscale": 0
}
```

This shows the different values for each image.

The left image was processed with the low pass filter, with a kernel size of 7 and a sigma of 2.0. The right image was processed with the high pass filter, with a kernel size of 3 and a sigma of 5.3. The mixin ratio was 0.55, or 55%.

The following JSON object is the correspondence used:

```json
{
  "first_image_points": [
    [
      86.0, 
      84.0
    ], 
    [
      49.0, 
      436.0
    ], 
    [
      280.0, 
      251.0
    ]
  ], 
  "second_image_points": [
    [
      80.0, 
      94.18244406196213
    ], 
    [
      63.87596899224806, 
      588.6402753872634
    ], 
    [
      419.84496124031006, 
      374.2512908777969
    ]
  ], 
  "first_image": "./left.png", 
  "second_image": "./right.png"
}
```
