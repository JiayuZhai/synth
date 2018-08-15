python experiment2.py -model WGAN_GP -hs 50 -visual False >>exp.txt
echo 'Training Finish'
python visual2.py -model WGAN_GP -hs 50 -visual False >>exp.txt
echo 'Evaluating Finish'
mv WGAN_GP/*.pdf random_log/50/
echo 'Moving Finish'

python experiment2.py -model WGAN_GP -hs 75 -visual False >>exp.txt
echo 'Training Finish'
python visual2.py -model WGAN_GP -hs 75 -visual False >>exp.txt
echo 'Evaluating Finish'
mv WGAN_GP/*.pdf random_log/75/
echo 'Moving Finish'

python experiment2.py -model WGAN_GP -hs 100 -visual False >>exp.txt
echo 'Training Finish'
python visual2.py -model WGAN_GP -hs 100 -visual False >>exp.txt
echo 'Evaluating Finish'
mv WGAN_GP/*.pdf random_log/100/
echo 'Moving Finish'
