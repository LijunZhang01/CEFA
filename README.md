# CEFA: A Plug-and-Play Method for Rare Human-Object Interactions Detection by Bridging Domain Gap

Code for our ACM MM24
paper "[A Plug-and-Play Method for Rare Human-Object Interactions
Detection by Bridging Domain Gap](http://arxiv.org/abs/2407.21438)"
.![image-20240805145200529](https://github.com/LijunZhang01/CEFA/blob/master/tmp/1212.png)

Contributed by Lijun Zhang, Wei Suo, Peng Wang, Yanning Zhang.

## Installation

Install the dependencies.

```
pip install -r requirements.txt
```

## Data preparation

### HICO-DET

HICO-DET dataset can be downloaded [here](https://drive.google.com/open?id=1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk). After
finishing downloading, unpack the tarball (`hico_20160224_det.tar.gz`) to the `data` directory.

### V-COCO

First clone the repository of V-COCO from [here](https://github.com/s-gupta/v-coco), and then follow the instruction to
generate the file `instances_vcoco_all_2014.json`. Next, download the prior file `prior.pickle`
from [here](https://drive.google.com/drive/folders/10uuzvMUCVVv95-xAZg5KS94QXm7QXZW4). 

## Training

### HICO-DET

```
# default setting
sh ./scripts/train.sh
```

## Acknowledge

Codes are built from [GEN-VLKT](https://github.com/YueLiao/gen-vlkt), [HOICLIP](https://github.com/Artanic30/HOICLIP)
, [DETR](https://github.com/facebookresearch/detr), [QPIC](https://github.com/hitachi-rd-cv/qpic) and [CDN](https://github.com/YueLiao/CDN). We thank them for their contributions.

# Release Schedule

- [ ] We will update more detailed README.md (including dataset, training, verification) in the future

    
