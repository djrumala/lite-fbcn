
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # see issue #152
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # -1 !!!!
import tensorflow as tf
from keras.layers import Dense, Lambda, Dropout, GlobalMaxPooling2D, Input, BatchNormalization, Reshape, Activation, Layer, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, GlobalAveragePooling2D
from keras.models import Model

class CustomPretrained(Model):
    def __init__(self, num_classes=5, img_shape=(224, 224, 3), base_model='mobilenet', model_name='B-MobileNets-P0', include_pooling='fast_bilinear', channel_reducer=640):
        super(CustomPretrained, self).__init__()  # Call the superclass's __init__ method
        self.include_pooling = include_pooling
        self.base_model = base_model
        self.num_classes = num_classes
        self.img_shape = img_shape
        self.channel_reducer = channel_reducer
        self.model_name = model_name

        if include_pooling:
            self.final_layers = [Dense(num_classes, activation='softmax', name='predictions')]
            if include_pooling == 'fast_bilinear':
                if "densenet" in base_model.lower():
                    self.backbone_model = tf.keras.applications.DenseNet121(input_shape=img_shape,include_top=False,weights='imagenet')
                elif "efficient" in self.base_model.lower():  
                    self.backbone_model = tf.keras.applications.EfficientNetB0(input_shape=img_shape,include_top=False,weights='imagenet')
                elif "mobilenetv2" in self.base_model.lower():
                    self.backbone_model = tf.keras.applications.MobileNetV2(input_shape=img_shape,include_top=False,weights='imagenet')
                elif "mobilenetv1" in self.base_model.lower():
                    self.backbone_model = tf.keras.applications.MobileNet(input_shape=img_shape,include_top=False,weights='imagenet')
            elif include_pooling == 'bilinear':
                if "mobilenetv1" in base_model.lower():
                    self.backbone_model_1 = tf.keras.applications.MobileNet(input_shape=img_shape,include_top=False,weights='imagenet')
                if "mobilenetv2" in base_model.lower():
                    self.backbone_model_2 = tf.keras.applications.MobileNetV2(input_shape=img_shape,include_top=False,weights='imagenet')

        else:
            self.final_layers = [Dropout(0.5),
                                 Dense(512, name='fc_1', activation='relu'),
                                 Dropout(0.5),
                                 Dense(256, name='fc_2', activation='relu'),
                                 Dense(num_classes, activation='softmax', name='predictions')]
            if "densenet" in base_model.lower():
                self.backbone_model =tf.keras.applications.DenseNet121(input_shape=img_shape, include_top=False, pooling='avg', weights='imagenet')
            elif "efficient" in base_model.lower():
                self.backbone_model =tf.keras.applications.EfficientNetB0(input_shape=img_shape, include_top=False, pooling='avg', weights='imagenet')
            elif "mobilenetv1" in base_model.lower():
                self.backbone_model =tf.keras.applications.MobileNet(input_shape=img_shape, include_top=False, pooling='avg', weights='imagenet')
            elif "mobilenetv2" in base_model.lower():
                self.backbone_model =tf.keras.applications.MobileNetV2(input_shape=img_shape, include_top=False, pooling='avg', weights='imagenet')
            
    def call(self, inputs):

        if self.include_pooling:
            if self.include_pooling == "fast_bilinear":
                x = inputs
                x = self.backbone_model(x)

                if self.channel_reducer: #(If None there is no channel reducer --> For the original fast bilinear, not lite-fbcn)
                    x = Conv2D(self.channel_reducer, (1, 1), activation='relu', name='conv_reduce_channels')(x)

                d1 = Dropout(0.5, name = 'dropout_1')(x)
                d2 = Dropout(0.5, name = 'dropout_2')(x)
                
                x = tf.keras.layers.Lambda(outer_product, name='outer_product')([d1,d2])
                x = tf.keras.layers.BatchNormalization()(x)

                for layer in self.final_layers:
                    x = layer(x)

            elif self.include_pooling == 'bilinear':
                x = inputs
            
                x1 = self.backbone_model_1(x)
                x2 = self.backbone_model_2(x)

                d1 = Dropout(0.5, name = 'dropout_1')(x1)
                d2 = Dropout(0.5, name = 'dropout_2')(x2)
                
                x = tf.keras.layers.Lambda(outer_product, name='outer_product')([d1,d2])
                x = tf.keras.layers.BatchNormalization()(x)

                for layer in self.final_layers:
                    x = layer(x)

        else:
            x = self.backbone_model.output
            for layer in self.final_layers:
                x = layer(x)

        return x
    
    def build_model(self):
        inputs = Input(shape=self.img_shape)
        outputs = self.call(inputs)
        if self.include_pooling:
            model = Model(inputs=inputs, outputs=outputs, name=self.model_name)
        else:
            model = Model(inputs=self.backbone_model.input, outputs=outputs, name=self.base_model)
        return model
  
def outer_product(x):    
    # print("Shape of x[0]:", x[0].shape)
    # print("Shape of x[1]:", x[1].shape)
    #Einstein Notation  [batch,1,1,depth] x [batch,1,1,depth] -> [batch,depth,depth]
    phi_I = tf.einsum('ijkm,ijkn->imn',x[0],x[1])
    # phi_I = tf.einsum('bi,bj->bij', x[0], x[1])

    # Reshape from [batch_size,depth,depth] to [batch_size, depth*depth]
    phi_I = tf.reshape(phi_I,[-1,x[0].shape[3]*x[1].shape[3]])
    
    # # Divide by feature map size [sizexsize]
    # size1 = int(x[1].shape[1])
    # size2 = int(x[1].shape[2])

    # Determine size1 and size2 dynamically
    size1 = tf.cast(tf.shape(x[1])[1], dtype=tf.float32)
    size2 = tf.cast(tf.shape(x[1])[2], dtype=tf.float32)

    phi_I = tf.divide(phi_I, size1*size2)
    
    # Take signed square root of phi_I
    y_ssqrt = tf.multiply(tf.sign(phi_I),tf.sqrt(tf.abs(phi_I)+1e-12))
    
    # Apply l2 normalization
    z_l2 = tf.nn.l2_normalize(y_ssqrt)
    return z_l2