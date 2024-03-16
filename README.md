# [Training A Small Emotional Vision Language Model for Visual Art Comprehension](https://arxiv.org/xxxx)
Official Code for SEVLM:  <br>
[arXiv](https://arxiv.org/xxxx) 
<br>

### Citation
If you find this work useful in your research, please consider citing:
```bash

```

### Requirements
- [PyTorch](https://pytorch.org/) 1.8 or higher
- `pip install git+https://github.com/openai/CLIP.git`
-  `pip install transformers`
-  `pip install git+https://github.com/huggingface/accelerate`

### Images Download
We conduct experiments on artEmis Dataset [WikiArt dataset](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset), and then resize images into a 600px resized folder, named `wikiart_rescaled_max_size_to_600px_same_aspect_ratio` 

### Annotations Download
You can dowloaded the structured annotations from [here](https://drive.google.com/drive/folders/1yMCYnEtqVFWCPoIeVflZDdiY7l5FP2HI?usp=sharing).
You also need [cococaption](https://github.com/tylin/coco-caption)  in the correct format in order to perform evaluation.


### Code
Please run from the command line with: <br>
```bash
accelerate launch artEmisX_train.py --ckpt_path /xxxx/ckpts/  --caption_save_path /xxxx/results/  --nle_data_train_path /xxxx/data/artEmis/artEmisX_cl_train.json  --nle_data_val_path /xxxx/data/artEmis/artEmisX_val.json
```


## Acknowledgement

[NLX-GPT](https://github.com/fawazsammani/nlxgpt): We are based on this code base, thanks for the open source!
