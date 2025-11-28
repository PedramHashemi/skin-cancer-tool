import data_loaders

mean, std = data_loaders.data_stats(
    data_dir="data/train",
    img_size=(224, 224)
)
print("Mean:", mean)
print("Standard Deviation:", std)