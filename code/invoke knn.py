Xtr,ytr,Xte,yte = load_CIFAR10('data//cifar10/')
Xtr_rows = Xtr.reshape(Xtr.shape[0],32*32*3)
Xte_rows = Xte.reshape(Xte.shape[0],32*32*3)
knn = NearestNeighbor()
nn.train(Xtr_rows,ytr)
yte_predict = nn.predict(Xte_rows)
print('accuracy %f',%(np.mean(Yte_predict == Yte)))