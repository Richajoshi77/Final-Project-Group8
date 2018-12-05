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

ayer_conv1, weights_conv1 = \
    new_conv_layer(input=x_image,
                   num_input_channels=3,
                   filter_size=filter_size1,
                   output_fm=num_filters1,
                   use_pooling=True)

layer_conv2, weights_conv2 = \
    new_conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   output_fm=num_filters2,
                   use_pooling=True)

layer_conv3, weights_conv3 =\
    new_conv_layer(input=layer_conv2,
                   num_input_channels=num_filters2,
                   filter_size=filter_size3,
                   output_fm=num_filters3,
                   use_pooling=True)

layer_conv4, weights_conv4 = \
    new_conv_layer(input=layer_conv3,
                   num_input_channels=num_filters3,
                   filter_size=filter_size4,
                   output_fm=num_filters4,
                   use_pooling=True)
layer_flat, num_features = flatten_layer(layer_conv4)

layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True,
                         use_dropout=True)

layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=num_classes,
                         use_relu=False,
                         use_dropout=False)

y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, axis=1)
y_true_cls= tf.argmax(y_true, axis=1)

# Transfer function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true)
