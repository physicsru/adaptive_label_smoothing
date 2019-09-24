for nr in 0.2 0.5 0.65 0.8 0.85
  for seed_num in 1 2 3 4 5 6 7 8 9 10
  do
    python vali_el.py --batch-size 128 --dataset mnist --noise_type symmetric --noise_rate ${nr} --n_epoch 150 --eps 9.99 --seed ${seed_num}
    python loss_el.py --batch-size 128 --dataset mnist --noise_type symmetric --noise_rate ${nr} --n_epoch 150 --eps 9.99 --seed ${seed_num}
  done
done
