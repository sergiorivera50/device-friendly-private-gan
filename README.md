# Device-Friendly Privacy-Preserving Generative Adversarial Networks

## Abstract

In an era where computational efficiency and data privacy are prominent among synthetic data generation techniques, 
this study introduces an end-to-end solution to tackle such complex issues: Device-Friendly Privacy-Preserving 
Generative Adversarial Networks (GANs). Merging the innovative techniques of GAN Slimming (GS) for model compression 
and Differential Privacy (DP) for data confidentiality, this research achieves fully converged GANs that are both 
lightweight and privacy-aware while retaining relatively accurate generations.

## How to use this Repository

### 1. Train the Teacher Model

To create a compressed GAN architecture, one has to first train a larger teacher model that will be later employed to
create a distillation dataset as part of the GAN Slimming process. Training the teacher generator is the first step for
this project.

```bash
python train_teacher.py --epochs 30 --batch-size 128
```

### 2. Create Distillation Dataset

Once fully trained, we make use of the converged generator to create a large dataset only containing generated samples.
These will be used on the next step to train smaller student models.

```bash
python create_dataset.py --teacher-dir <MODEL_PATH> --num-samples 200000
```

### 3. Train Student Generators

Before conducting training on a given student model, first change the quantisation level at the `models/__init__.py`
file. There you can specify the number of bits for the quantisation part of the GAN Slimming framework. Once selected, 
simply run the student training script to begin the distillation, channel pruning and quantisation process.

```bash
python train_student.py --epochs 20 --batch-size 128 --rho 0.01
```

### 4. Compress Student Models

Reaching convergence with a given student generator is not sufficient to truly appreciate the reduction in computational
and memory size. One has to actually select the non-zeroed significant channels (resulting from the GS process) and copy
their weights into a newly formed smaller generator architecture. Fortunately, we provide a utility to seamlessly perform
this process, just remember to read `compress_student.py` and specify the model path for your converged student model.

```bash
python compress_student.py
```

### 5. Fine-tune with Differential Privacy

Finally, you can take your compressed slimmed generator and fine-tune it with differential privacy mechanisms, this will
give you a final fully compressed, privacy-preserving GAN. Afterwards, you can simply discard the discriminator model and
deploy your generator into a resource-constrained device.

Note: modify in `fine_tune.py` the model path to point towards your slimmed generator.

```bash
python fine_tune.py --epochs 20 --batch-size 128
```

