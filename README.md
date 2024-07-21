<img src="https://yuheng.ink/project-page/pyramid-discrete-diffusion/images/pyramid_logo.png" height="70px" align="left">

# Pyramid Diffusion for Fine 3D Large Scene Generation

[Yuheng Liu](https://yuheng.ink/)<sup>1,2</sup>, [Xinke Li](https://shinke-li.github.io/)<sup>3</sup>, [Xueting Li](https://sunshineatnoon.github.io/)<sup>4</sup>, [Lu Qi](http://luqi.info/)<sup>5</sup>, [Chongshou Li](https://scholar.google.com.sg/citations?user=pQsr70EAAAAJ&hl=en)<sup>1</sup>, [Ming-Hsuan Yang](https://scholar.google.com/citations?user=p9-ohHsAAAAJ&hl=en&oi=ao)<sup>5,6</sup>

<sup>1</sup>Southwest Jiaotong University, <sup>2</sup>University of Leeds, <sup>3</sup>City University of HongKong, <sup>4</sup>NVIDIA, <sup>5</sup>The University of Cailfornia, Merced, <sup>6</sup>Yonsei University

![Endpoint Badge](https://img.shields.io/endpoint?url=https%3A%2F%2Fhits.dwyl.com%2FYuheng-SWJTU%2Fpyramid-discrete-diffusion.json&label=visitors&color=fedcba)  [![Static Badge](https://img.shields.io/badge/PDF-Download-red?logo=Adobe%20Acrobat%20Reader)](https://yuheng.ink/project-page/pyramid-discrete-diffusion/papers/Pyramid_Diffusion_for_Fine_3D_Large_Scene_Generation.pdf)  [![Static Badge](https://img.shields.io/badge/2311.12085-b31b1b?logo=arXiv&label=arXiv)](https://arxiv.org/abs/2311.12085)  [![Static Badge](https://img.shields.io/badge/Project%20Page-blue?logo=Google%20Chrome&logoColor=white)](https://yuheng.ink/project-page/pyramid-discrete-diffusion/) 
<!-- [![Static Badge](https://img.shields.io/badge/Youtube-%23ff0000?style=flat&logo=Youtube)](https://www.youtube.com/watch?v=g4fleCzy4EI) -->

![Teaser](https://yuheng.ink/project-page/pyramid-discrete-diffusion/images/teaser.png)

Diffusion models have shown remarkable results in generating 2D images and small-scale 3D objects. However, their application to the synthesis of large-scale 3D scenes has been rarely explored. This is mainly due to the inherent complexity and bulky size of 3D scenery data, particularly outdoor scenes, and the limited availability of comprehensive real-world datasets, which makes training a stable scene diffusion model challenging. In this work, we explore how to effectively generate large-scale 3D scenes using the coarse-to-fine paradigm. We introduce a framework, the Pyramid Discrete Diffusion model (PDD), which employs scale-varied diffusion models to progressively generate high-quality outdoor scenes. Experimental results of PDD demonstrate our successful exploration in generating 3D scenes both unconditionally and conditionally. We further showcase the data compatibility of the PDD model, due to its multi-scale architecture: a PDD model trained on one dataset can be easily fine-tuned with another dataset.

## NEWS

- [2024/07/02] 🎉 Our work is accepted by ECCV 24.
- [2023/11/22] Our work is now on [arXiv](https://arxiv.org/abs/2311.12085).
- [2023/11/20] Official repo is created, code will be released soon, access our [Project Page](https://yuheng.ink/project-page/pyramid-discrete-diffusion/) for more details.

