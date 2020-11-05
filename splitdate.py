index = np.arange(len(dataset))
from sklearn.model_selection import train_test_split
X_train, X_test= train_test_split(index, test_size=0.2, random_state=42)
X_train.shape,X_test.shape
label_txt = np.load("pascal_label_txt.npy")
embedding_img = np.load("pascal_embedding_img.npy")
embedding_txt = np.load("pascal_embedding_txt.npy")
label_txt = np.expand_dims(label_txt, axis=1)
img_tra,img_tes = embedding_img[X_train], embedding_img[X_test]
txt_tra,txt_tes = embedding_txt[X_train], embedding_txt[X_test]
lab_tra,lab_tes = label_txt[X_train], label_txt[X_test]
print(img_tra.shape, img_tes.shape)
print(txt_tra.shape, txt_tes.shape)
print(lab_tra.shape, lab_tes.shape)
print(lab_tes)
import scipy.io as scio
path = '/home/hammer/DSCMR-master/data/pascal/'
scio.savemat(path+"train_img.mat", {'train_img':img_tra})
scio.savemat(path+"test_img.mat", {'test_img':img_tes})
scio.savemat(path+"train_txt.mat", {'train_txt':txt_tra})
scio.savemat(path+"test_txt.mat", {'test_txt':txt_tes})
scio.savemat(path+"train_img_lab.mat", {'train_img_lab':lab_tra})
scio.savemat(path+"test_img_lab.mat", {'test_img_lab':lab_tes})
