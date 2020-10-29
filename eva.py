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

def __getitem__(self, index):
        img = cv2.imread('/home/w00536717/hammer/DSCMR-master/data/flickr30k_images/flickr30k_images/'+ self.name[index])
        print('/home/w00536717/hammer/DSCMR-master/data/flickr30k_images/flickr30k_images/'+ self.name[index])
        img1 = cv2.rotate(img, cv2.ROTATE_180)
        img2 = cv2.GaussianBlur(img,(3,3),0)
        img3 = cv2.GaussianBlur(img,(1,1),0)
        img4 = cv2.GaussianBlur(img,(5,5),0)
        img = cv2.resize(img, self.dim, interpolation = cv2.INTER_AREA)
        img1 = cv2.resize(img1, self.dim, interpolation = cv2.INTER_AREA)
        img2 = cv2.resize(img2, self.dim, interpolation = cv2.INTER_AREA)
        img3 = cv2.resize(img3, self.dim, interpolation = cv2.INTER_AREA)
        img4 = cv2.resize(img4, self.dim, interpolation = cv2.INTER_AREA)
        out = np.concatenate((img, img1), axis=0)
        out = np.concatenate((out, img2), axis=0)
        out = np.concatenate((out, img3), axis=0)
        out = np.concatenate((out, img4), axis=0)
        out = np.array([img,img1,img2,img3,img4][:],dtype=np.float32)#torch.FloatTensor([img,img1,img2,img3,img4])
        return torch.from_numpy(out).float(), self.label_img[index]
