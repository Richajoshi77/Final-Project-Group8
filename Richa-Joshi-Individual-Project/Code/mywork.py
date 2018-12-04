number_images_shape= 0
dim= (60,60)

for i in os.listdir(dir):
    name = os.path.basename(i)
    l = os.path.splitext(name)[0]
    img = cv2.imread(i, cv2.IMREAD_UNCHANGED)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    if resized.shape == (60, 60, 3):
        a = np.array(resized)
        allImage[p - 1] = a
        p = p + 1
        x = labels.loc[labels["ImageID"] == l, "Class"]
        Class_order.append(x.values[0])

print(number_images_shape)

#One hot encoding
encoder = LabelEncoder()
encoder.fit(class_labels)
encoded_y= encoder.transform(class_labels)
dummy_y = np_utils.to_categorical(encoded_y)
print (dummy_y.shape)


