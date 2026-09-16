# v5 semi-synthetic alignment benchmark

The pre image is transformed by a saved known affine matrix, then positive Gaussian spikes are injected into the moving image. All methods see the same moving image and truth locations. Recovery is truth points matched within 3 px after inverse warping; false-positive rate is unmatched detections divided by all detections.
