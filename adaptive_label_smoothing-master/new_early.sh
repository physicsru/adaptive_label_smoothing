for nr in 0.2 0.5 0.65 0.8 0.85
  do
  for seed_num in 1 2 3
  do
    python vali_el.py --batch-size 128 --dataset mnist --noise_type symmetric --noise_rate ${nr} --n_epoch 50 --eps 9.99 --seed ${seed_num}
  done
done
