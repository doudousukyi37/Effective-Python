import h5py

# 列出HDF5文件中存储的所有权重数据，包括它们的名称和对应的值
f = h5py.File('./save_weights/resNet_1.h5', 'r')
for root_name, g in f.items():
    print(root_name)
    for _, weights_dirs in g.attrs.items():
        for i in weights_dirs:
            name = root_name + "/" + str(i, encoding="utf-8")
            data = f[name]
            print(data.value)






