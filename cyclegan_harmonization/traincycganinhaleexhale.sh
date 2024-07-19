#!/bin/bash

python traincyclegan.py --dataroot /home-local/krishar1/Inhale_Exhale_CT/train_slices --source_kernel inspiratory_BONE --target_kernel expiratory_STANDARD --name inspiration_expiration_COPD --model vanillacycle_gan --norm instance --no_dropout --netG resnet_9blocks --dataset_mode unaligned --display_id 0 --gpu_ids 0 --batch_size 6 --load_size 512 --crop_size 512 --no_flip --num_threads 6 --n_epochs 25 --n_epochs_decay 25
