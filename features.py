from musicnn.extractor import extractor


file_name = 'songs/pop.00000.wav'

#model = 'MSD_musicnn_big'
model = 'MSD_vgg'

taggram, tags, features = extractor(file_name, model=model, extract_features=True)

print(features.keys())
