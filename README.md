# Training A Small Emotional Vision Language Model for Visual Art Comprehension
Official Code for SEVLM:  [arXiv](https://arxiv.org/abs/2403.11150) 
<br>


### Requirements
- [PyTorch](https://pytorch.org/) 1.8 or higher
- `pip install git+https://github.com/openai/CLIP.git`
-  `pip install transformers`
-  `pip install git+https://github.com/huggingface/accelerate`

### Images Download
We conduct experiments on artEmis Dataset [WikiArt dataset](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset), and then resize images into a 600px resized folder, named `wikiart_rescaled_max_size_to_600px_same_aspect_ratio` 

### Data preprocessing
We conduct experiments on two benchmark datasets: [ArtEmis v1.0 ](https://arxiv.org/abs/2101.07396)  and [ArtEmis v2.0](https://arxiv.org/abs/2204.07660). You need to perform [data preprocessing](preprocess_data.md)  for both datasets. 


### Code
Please run from the command line with: <br>
```bash
accelerate launch artEmisX_train.py --ckpt_path /xxxx/ckpts/  --caption_save_path /xxxx/results/  --nle_data_train_path /xxxx/data/artEmis/artEmisX_cl_train.json  --nle_data_val_path /xxxx/data/artEmis/artEmisX_val.json
```

### Citation
If you find this work useful in your research, please consider citing:
```
@misc{zhang2024training,
      title={Training A Small Emotional Vision Language Model for Visual Art Comprehension}, 
      author={Jing Zhang and Liang Zheng and Dan Guo and Meng Wang},
      year={2024},
      eprint={2403.11150},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgement

- We thank [NLX-GPT](https://github.com/fawazsammani/nlxgpt) for  open-source implementation of their language model. SEVLM repo is built on [NLX-GPT](https://github.com/fawazsammani/nlxgpt).
