for seed_num in 1 2 3 4 5 6 7 8 9 10
do
  python vali_el.py --batch-size 128 --dataset mnist --noise_type symmetric --noise_rate 0.5 --n_epoch 150 --eps ${n} --seed ${seed_num}
  python loss_el.py --batch-size 128 --dataset mnist --noise_type symmetric --noise_rate 0.5 --n_epoch 150 --eps ${n} --seed ${seed_num}
done
