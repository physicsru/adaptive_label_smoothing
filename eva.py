def fx_calc_map_label(image, text, label, k = 0, dist_method='COS'):
  if dist_method == 'L2':
    dist = scipy.spatial.distance.cdist(image, text, 'euclidean')
  elif dist_method == 'COS':
    dist = scipy.spatial.distance.cdist(image, text, 'cosine')
  ord = dist.argsort()
  numcases = dist.shape[0]
  res = []
  for i in range(numcases):
    order = ord[i]
    p = 0.0
    r = 0.0
    for j in range(numcases):
      if label[i] == label[order[j]]:
          res += [j]
          break
  rank = [1, 5, 10]
  acc = [sum([_ < r for _ in res]) / len(res) for r in rank]
  return acc[0]
