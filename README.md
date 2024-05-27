# Resolution Enhancement of Low-Quality Images

## Overview

This repository contains a Jupyter Notebook for training a model designed to enhance the quality of images. By leveraging Generative Adversarial Networks (GANs), specifically the Super-Resolution GAN (SRGAN) architecture, this project aims to improve image resolution for various applications, ranging from surveillance to diverse industrial and research uses.

## Project Significance

High-quality images are crucial across numerous industries for tasks such as crime investigation, activity monitoring, and overall security enforcement. Poor image quality often hampers the accurate identification of key elements. Enhancing image resolution can revolutionize visual data applications across various fields, augmenting the quality and utility of images for diverse purposes.

## Methodology

### SRGAN Architecture

The SRGAN methodology utilizes GANs to enhance image resolution. The SRGAN architecture comprises two main components:
- **Generator:** A deep Convolutional Neural Network (CNN) that upscales low-resolution images.
- **Discriminator:** A network that distinguishes between real and generated high-resolution images.

Adversarial training iteratively refines both networks, guided by perceptual and adversarial loss functions.

### Training Data

The model is trained on a diverse set of datasets, including:
- **MIRFLICKR-25000:** A collection of 25,000 images from Flickr, used for image analysis and tagging research.
- **Flickr-Faces-HQ (FFHQ):** Over 70,000 high-quality human face images, essential for facial recognition and image generation.
- **Human Detection Dataset:** CCTV footage for human detection and tracking.
- **ImageNet:** Approximately 14 million images spanning numerous categories for comprehensive object recognition.
- **Intel Images:** 25,000 images of outdoor scenes, enriching the model's scene classification and object detection capabilities.

### Training Process

Images are preprocessed by resizing to standardized resolutions of 32x32 or 64x64 and organized into paired low and high-resolution sets. The training process includes:

1. **Initial Training:** On MIRFLICKR-1M with a subset of 5,000 images.
2. **Refined Training:** Extended training epochs (20-40) and strategic dataset adjustments led to a four-fold improvement in image quality.

During each epoch:
- The discriminator is trained with real and fake images, distinguishing between them using real (1) and fake (0) labels.
- The generator is trained with low-resolution images, high-resolution images, real labels, and image features extracted from high-resolution images using VGG19.

### Model Evolution

The project has undergone several iterations, with key phases including:
- Initial training with 5,000 images over 10 epochs.
- Dataset augmentation and hyperparameter adjustments.
- Introduction of Enhanced Super-Resolution GAN (ESRGAN).
- Final breakthrough with a 20-epoch model, transforming low-resolution inputs to 128x128 images.
- Further refinement using the Flickr-Faces-HQ dataset.

## Contents

- `model-training.ipynb`: The Jupyter Notebook containing the model training code.
- `research_paper.pdf`: A detailed research paper pre-print documenting the project's methodology, datasets, training process, and results.

## Getting Started

To get started with this project, clone the repository and follow the instructions in the Jupyter Notebook to set up the environment and begin training the model.

```bash
git clone https://github.com/yourusername/Image-Enhancement.git
cd Image-Enhancement
```

Open the Jupyter Notebook to explore the code and start the training process.

## Contributing

We welcome contributions to improve the project. Feel free to submit pull requests or open issues for any bugs or enhancement suggestions.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

---

We hope this project provides valuable insights and tools for enhancing images across various applications. Happy coding!

---

For any questions or further information, please contact hasnain.ai3142@gmail.com
