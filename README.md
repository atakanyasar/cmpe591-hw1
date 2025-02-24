# CmpE 591 - HW1 ATAKAN YAŞAR - RAMAZAN ONUR ACAR

## !!! Our hw1_1.pt is more than 25 mb, so it is over the limits of a file to upload to github. It can be downloaded from the google drive with the link
### https://drive.google.com/file/d/1cOGweDkN0hHrY7irN5pD_kNqd_Bf7L5H/view?usp=sharing
### It is also available in "hw1_1.pt.txt"

---

In Part 1 and Part 2, we used different optimizers and model architectures to compare performance.

## Part 1: MLP 

We used the Adam and SGD optimizers and compared their performance.

### Loss Plot

The training and validation loss for the MLP model with Adam and SGD optimizers are shown below:

- **Adam:** 

  
  ![MLP Training Average - Adam](https://raw.githubusercontent.com/atakanyasar/cmpe591-hw1/refs/heads/main/training_history_mlp_adam_average.png)
  
  ![MLP Training Folds - Adam](https://raw.githubusercontent.com/atakanyasar/cmpe591-hw1/refs/heads/main/training_history_mlp_adam_folds.png)
  
- **SGD:**

  ![MLP Training Average - SGD](https://raw.githubusercontent.com/atakanyasar/cmpe591-hw1/refs/heads/main/training_history_mlp_sgd_average.png)

  ![MLP Training Folds - SGD](https://raw.githubusercontent.com/atakanyasar/cmpe591-hw1/refs/heads/main/training_history_mlp_sgd_folds.png)

#### Conclusion:
- The **Adam optimizer** performed better in terms of faster convergence and lower final validation loss compared to **SGD**. We used Adam for the final model.

---

## Part 2: CNN

We compared the performance of Adam and SGD optimizers in this part, too.

### Loss Plot

The training and validation loss for the CNN model with Adam and SGD optimizers are shown below:

- **Adam:**
  
  ![CNN Training Average - Adam](https://raw.githubusercontent.com/atakanyasar/cmpe591-hw1/refs/heads/main/training_history_cnn_adam_average.png)
  
  ![CNN Training Folds - Adam](https://raw.githubusercontent.com/atakanyasar/cmpe591-hw1/refs/heads/main/training_history_cnn_adam_folds.png)
  
- **SGD:**

  ![CNN Training Average - SGD](https://raw.githubusercontent.com/atakanyasar/cmpe591-hw1/refs/heads/main/training_history_cnn_sgd_average.png)

  ![CNN Training Folds - SGD](https://raw.githubusercontent.com/atakanyasar/cmpe591-hw1/refs/heads/main/training_history_cnn_sgd_folds.png)

#### Conclusion:
- **Adam optimizer** again showed better performance, achieving better convergence and lower final validation loss than **SGD**. Thus, we chose Adam for the final CNN model.

---

## Part 3: Reconstruction

We used a Variational Autoencoder (VAE) model with a latent space to reconstruct the image.

VAE is a deep learning model designed for generative tasks, where it learns to compress an input into a latent space and then reconstruct it back to the original form. In our case, the VAE takes the image and the action (encoded as a one-hot vector) as input and reconstructs the resulting image. The model consists of an encoder, which maps the input image and action to a probabilistic latent space, and a decoder, which reconstructs the output image from this latent representation. The VAE was trained with a combination of reconstruction loss and KL divergence loss . The training used the Adam optimizer.

### Loss Plot

The training and validation loss for the VAE model during training are shown below:


- **Test loss** showed good convergence, suggesting that the VAE generalizes well to unseen data.

  ![VAE Losses](https://raw.githubusercontent.com/atakanyasar/cmpe591-hw1/refs/heads/main/vae_losses.png)

#### Image Reconstruction Example

![VAE Results](https://raw.githubusercontent.com/atakanyasar/cmpe591-hw1/refs/heads/main/vae_results.png)


#### Conclusion:
- The **VAE model** was able to generate reasonable reconstructions of the environment after an action, demonstrating its ability to learn the latent space and generate accurate predictions.

---


The final trained models have been saved as `hw1_1.pt.txt`, `hw1_2.pt`, and `hw1_3.pt`. The trained models can be loaded for testing novel inputs as demonstrated in the test method.

---

## Model Files

- [hw1_1.pt.txt](https://github.com/atakanyasar/cmpe591-hw1/blob/main/hw1_1.pt.txt)
- [hw1_2.pt](https://github.com/atakanyasar/cmpe591-hw1/blob/main/hw1_2.pt)
- [hw1_3.pt](https://github.com/atakanyasar/cmpe591-hw1/blob/main/hw1_3.pt)
