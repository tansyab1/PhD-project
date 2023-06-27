# NAT - Object Detection and Instance Segmentation

Make sure to set up your environment according to the [object detection README](README.md).

## Training on COCO

### Mask R-CNN
<details>
<summary>
<b>NAT-Mini + Mask R-CNN</b>
</summary>

```shell
./dist_train.sh configs/nat/mask_rcnn_nat_mini_3x_coco.py $NUM_GPUS --cfg-options data.samples_per_gpu=$((16/$NUM_GPUS)) data.workers_per_gpu=$((16/$NUM_GPUS))
```
</details>
<details>
<summary>
<b>NAT-Tiny + Mask R-CNN</b>
</summary>

```shell
./dist_train.sh configs/nat/mask_rcnn_nat_tiny_3x_coco.py $NUM_GPUS --cfg-options data.samples_per_gpu=$((16/$NUM_GPUS)) data.workers_per_gpu=$((16/$NUM_GPUS))
```
</details>
<details>
<summary>
<b>NAT-Small + Mask R-CNN</b>
</summary>

```shell
./dist_train.sh configs/nat/mask_rcnn_nat_small_3x_coco.py $NUM_GPUS --cfg-options data.samples_per_gpu=$((16/$NUM_GPUS)) data.workers_per_gpu=$((16/$NUM_GPUS))
```
</details>

### Cascade Mask R-CNN
<details>
<summary>
<b>NAT-Mini + Cascade R-CNN</b>
</summary>

```shell
./dist_train.sh configs/nat/cascade_mask_rcnn_nat_mini_3x_coco.py $NUM_GPUS --cfg-options data.samples_per_gpu=$((16/$NUM_GPUS)) data.workers_per_gpu=$((16/$NUM_GPUS))
```
</details>
<details>
<summary>
<b>NAT-Tiny + Mask R-CNN</b>
</summary>

```shell
./dist_train.sh configs/nat/cascade_mask_rcnn_nat_tiny_3x_coco.py $NUM_GPUS --cfg-options data.samples_per_gpu=$((16/$NUM_GPUS)) data.workers_per_gpu=$((16/$NUM_GPUS))
```
</details>
<details>
<summary>
<b>NAT-Small + Mask R-CNN</b>
</summary>

```shell
./dist_train.sh configs/nat/cascade_mask_rcnn_nat_small_3x_coco.py $NUM_GPUS --cfg-options data.samples_per_gpu=$((16/$NUM_GPUS)) data.workers_per_gpu=$((16/$NUM_GPUS))
```
</details>
<details>
<summary>
<b>NAT-Base + Cascade Mask R-CNN</b>
</summary>

```shell
./dist_train.sh configs/nat/cascade_mask_rcnn_nat_base_3x_coco.py $NUM_GPUS --cfg-options data.samples_per_gpu=$((16/$NUM_GPUS)) data.workers_per_gpu=$((16/$NUM_GPUS))
```
</details>

## Validation
### Mask R-CNN
<details>
<summary>
<b>NAT-Mini + Mask R-CNN</b>
</summary>

```shell
./dist_test.sh \
    configs/nat/mask_rcnn_nat_mini_3x_coco.py \
    https://shi-labs.com/projects/nat/checkpoints/DET/nat_mini_maskrcnn.pth \
    $NUM_GPUS \
    --eval bbox segm
```
</details>
<details>
<summary>
<b>NAT-Tiny + Mask R-CNN</b>
</summary>

```shell
./dist_test.sh \
    configs/nat/mask_rcnn_nat_tiny_3x_coco.py \
    https://shi-labs.com/projects/nat/checkpoints/DET/nat_tiny_maskrcnn.pth \
    $NUM_GPUS \
    --eval bbox segm
```
</details>
<details>
<summary>
<b>NAT-Small + Mask R-CNN</b>
</summary>

```shell
./dist_test.sh \
    configs/nat/mask_rcnn_nat_small_3x_coco.py \
    https://shi-labs.com/projects/nat/checkpoints/DET/nat_small_maskrcnn.pth \
    $NUM_GPUS \
    --eval bbox segm
```
</details>

### Cascade Mask R-CNN
<details>
<summary>
<b>NAT-Mini + Cascade R-CNN</b>
</summary>

```shell
./dist_test.sh \
    configs/nat/cascade_mask_rcnn_nat_mini_3x_coco.py \
    https://shi-labs.com/projects/nat/checkpoints/DET/nat_mini_cascademaskrcnn.pth \
    $NUM_GPUS \
    --eval bbox segm
```
</details>
<details>
<summary>
<b>NAT-Tiny + Cascade Mask R-CNN</b>
</summary>

```shell
./dist_test.sh \
    configs/nat/cascade_mask_rcnn_nat_tiny_3x_coco.py \
    https://shi-labs.com/projects/nat/checkpoints/DET/nat_tiny_cascademaskrcnn.pth \
    $NUM_GPUS \
    --eval bbox segm
```
</details>
<details>
<summary>
<b>NAT-Small + Cascade Mask R-CNN</b>
</summary>

```shell
./dist_test.sh \
    configs/nat/cascade_mask_rcnn_nat_small_3x_coco.py \
    https://shi-labs.com/projects/nat/checkpoints/DET/nat_small_cascademaskrcnn.pth \
    $NUM_GPUS \
    --eval bbox segm
```
</details>
<details>
<summary>
<b>NAT-Base + Cascade Mask R-CNN</b>
</summary>

```shell
./dist_test.sh \
    configs/nat/cascade_mask_rcnn_nat_base_3x_coco.py \
    https://shi-labs.com/projects/nat/checkpoints/DET/nat_base_cascademaskrcnn.pth \
    $NUM_GPUS \
    --eval bbox segm
```
</details>

## Checkpoints
| Backbone | Network | # of Params | FLOPs | mAP | Mask mAP | Checkpoint | Config |
|---|---|---|---|---|---|---|---|
| NAT-Mini | Mask R-CNN | 40M | 225G | 46.5 | 41.7 | [Download](https://shi-labs.com/projects/nat/checkpoints/DET/nat_mini_maskrcnn.pth) | [config.py](configs/nat/mask_rcnn_nat_mini_3x_coco.py) |
| NAT-Tiny | Mask R-CNN | 48M | 258G | 47.7 | 42.6 | [Download](https://shi-labs.com/projects/nat/checkpoints/DET/nat_tiny_maskrcnn.pth) | [config.py](configs/nat/mask_rcnn_nat_tiny_3x_coco.py) |
| NAT-Small | Mask R-CNN | 70M | 330G | 48.4 | 43.2 | [Download](https://shi-labs.com/projects/nat/checkpoints/DET/nat_small_maskrcnn.pth) | [config.py](configs/nat/mask_rcnn_nat_small_3x_coco.py) |
| NAT-Mini | Cascade Mask R-CNN | 77M | 704G | 50.3 | 43.6 | [Download](https://shi-labs.com/projects/nat/checkpoints/DET/nat_mini_cascademaskrcnn.pth) | [config.py](configs/nat/cascade_mask_rcnn_nat_mini_3x_coco.py) |
| NAT-Tiny | Cascade Mask R-CNN | 85M | 737G | 51.4 | 44.5 | [Download](https://shi-labs.com/projects/nat/checkpoints/DET/nat_tiny_cascademaskrcnn.pth) | [config.py](configs/nat/cascade_mask_rcnn_nat_tiny_3x_coco.py) |
| NAT-Small | Cascade Mask R-CNN | 108M | 809G | 52.0 | 44.9 | [Download](https://shi-labs.com/projects/nat/checkpoints/DET/nat_small_cascademaskrcnn.pth) | [config.py](configs/nat/cascade_mask_rcnn_nat_small_3x_coco.py) |
| NAT-Base | Cascade Mask R-CNN | 147M | 931G | 52.3 | 45.1 | [Download](https://shi-labs.com/projects/nat/checkpoints/DET/nat_base_cascademaskrcnn.pth) | [config.py](configs/nat/cascade_mask_rcnn_nat_base_3x_coco.py) |


