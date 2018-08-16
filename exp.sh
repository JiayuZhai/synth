# Run 3 GAN model training and data generation.
python experiment.py -model WGAN_GP -hs 50 >>exp.txt
echo 'Training Finish'
python visual.py -model WGAN_GP -hs 50 -method GAN >>exp.txt
echo 'Evaluating Finish'
mv data/GAN.npy data/GAN50.npy
echo 'Moving Finish'

python experiment.py -model WGAN_GP -hs 75 >>exp.txt
echo 'Training Finish'
python visual.py -model WGAN_GP -hs 75 -method GAN >>exp.txt
echo 'Evaluating Finish'
mv data/GAN.npy data/GAN75.npy
echo 'Moving Finish'

python experiment.py -model WGAN_GP -hs 100 >>exp.txt
echo 'Training Finish'
python visual.py -model WGAN_GP -hs 100 -method GAN >>exp.txt
echo 'Evaluating Finish'
mv data/GAN.npy data/GAN100.npy
echo 'Moving Finish'

# Run another two synthesis method.
python visual.py -method EXCHANGE >>exp.txt
python visual.py -method CART >>exp.txt

# Run the comparison for evaluations.
python visual.py -method compare >>exp.txt
