import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # or "1" depending on the GPU you want to use

import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.models import Model
from keras.utils import to_categorical
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.manifold import TSNE
import umap.umap_ as umap
from sklearn.utils import shuffle
import seaborn as sns

import itertools

from time import process_time
from datetime import datetime

from data_load import get_data_array
from model import CustomPretrained, CustomSFCN, StandardModel
# from numba import cuda

def model_evaluate(test_X, test_Y, cnn_model='mobilenetv1', sampling_name='None', input_shape=(224, 224, 3), num_class=5, optimizer='sgd', learning_rate=0.01, opt_nesterov=True):
    test_X = np.array(test_X)
    test_Y = to_categorical(test_Y)
   
    if optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.5, nesterov=opt_nesterov, name="SGD")
    elif optimizer =='adam':
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,  # Starting learning rate (adjust as needed)
            beta_1=0.9,           # Exponential decay rate for the 1st moment estimates
            beta_2=0.999,         # Exponential decay rate for the 2nd moment estimates
            epsilon=1e-07,        # Small constant for numerical stability
            decay=1e-5             # Learning rate decay (adjust as needed)
        )

    path = "../temp/checkpoint"
    checkpoints = []

    for checkpoint in os.listdir(path)[:]:
        if checkpoint.endswith(".h5"):
            cp_sample = checkpoint.split("_")[2] #to split sampling name (i.e. R1-final)
            cp_sample = cp_sample.split(".")[0]
            # print(checkpoint)

            if cnn_model.lower() in checkpoint.lower():
                # print(checkpoint)
                if sampling_name.lower() in cp_sample.lower():
                    print("Chosen:" + checkpoint)
                    checkpoints.append(checkpoint)
    
    for checkpoint in checkpoints:
        print(checkpoint)
        model_name = checkpoint.split('_')[1]
        model_name = model_name.split('-')[0]
        if "b-" in checkpoint.lower():
            if "fb-" in checkpoint.lower():
                include_pooling = "fast_bilinear"
            else:
                include_pooling = "bilinear"
        else:
            include_pooling = None

            available_backbones = ["MobileNetV1", "MobileNetV2", "DenseNet121", "EfficientNetB0"]
               
        for base_model in available_backbones:
            print(checkpoint.lower())
            print(base_model.lower())
            if base_model.lower() in checkpoint.lower():
                print(f"Configuring model with base_model: {base_model} for {checkpoint}")

                if '-P0' in checkpoint:
                    channel_reducer = None
                elif '-P2' in checkpoint:
                    channel_reducer = 640
                elif '-P4' in checkpoint:
                    channel_reducer = 320
                elif '-P8' in checkpoint:
                    channel_reducer = 160
                custom_model = CustomPretrained(base_model=base_model, model_name=model_name, num_classes=num_class, img_shape=input_shape, include_pooling=include_pooling, channel_reducer=channel_reducer)
                # model.build((None, 224, 224, 3))  # Pass an example input shape to build the model
                model = custom_model.build_model()
                model.summary()
            else:
                print("Model not found")   

        model.load_weights(f'../temp/checkpoint/{checkpoint}')

        tf.keras.backend.clear_session()
        tf.random.set_seed(51)
        np.random.seed(51)

        model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
        
        y_pred = model.predict(test_X) #make prediction
        
        CATEGORIES = ["CN", "NEO", "CVA", "NDD", "INF"]
        cm = confusion_matrix(np.argmax(test_Y, axis=1), np.argmax(y_pred, axis=1))
        
        if 'hold_out' in sampling_name:
            saved_title = f'../temp/test_log/cm/cm-ablation_{checkpoint}.png'
        else:
            saved_title = f'../temp/test_log/cm/cm-{checkpoint}.png'

        fig_title= "Testing Confusion Matrix\n"+ f'{base_model} with {include_pooling} pooling' 
        plot_confusion_matrix(cm, CATEGORIES, title = fig_title, save_as = saved_title)
        
        report = classification_report(np.argmax(test_Y, axis=1), np.argmax(y_pred, axis=1), target_names=CATEGORIES, digits=4)
        
        # print(report)
        # df = classification_report_csv(report)

        if 'hold_out' in sampling_name:
            filename = f'../temp/test_log/report/report-ablation_{checkpoint}.txt'
        else:
            filename = f'../temp/test_log/report/report-{checkpoint}.txt'
        # with open(filename, mode='w') as f:
        #     df.to_csv(f)
        save_report_txt(report, filename)

        

        if 'hold_out' in sampling_name:
            filename = f'../temp/test_log/umap/umap-ablation_{checkpoint}.txt'
        else:
            filename = f'../temp/test_log/report/umap-{checkpoint}.txt'

        plot_umap(model, test_X, test_Y, filename, 'predictions')
        

def model_build_train(train_X, train_Y, valid_X, valid_Y, round_label='1', input_shape=(224, 224, 3), num_class=5, num_ep=100, batch_size=64, optimizer='adam', learning_rate=0.01, opt_nesterov=True, lr_reduce=True, lr_reduce_factor=0.1, lr_reduce_patience=10, lr_min=0.00001):

    print("Whether or not to use ReduceonPlateau -->" + str(lr_reduce))

    train_X = np.array(train_X)
    valid_X = np.array(valid_X)

    train_Y = to_categorical(train_Y, num_class)
    valid_Y = to_categorical(valid_Y, num_class)
    
    #DEFINE OPTIMIZER
    if optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.5, nesterov=opt_nesterov, name="SGD")
    elif optimizer =='adam':
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate,  # Starting learning rate (adjust as needed)
            beta_1=0.9,           # Exponential decay rate for the 1st moment estimates
            beta_2=0.999,         # Exponential decay rate for the 2nd moment estimates
            epsilon=1e-07,        # Small constant for numerical stability
            decay=1e-5             # Learning rate decay (adjust as needed)
        )
        
    print(optimizer)

    pretrained_configs = [
                ["mobilenetv2", None, "MobileNetV2", None],
                ["mobilenetv2", "fast_bilinear", "FB-MobileNetV2-P4", 320],
                ["mobilenetv2", "fast_bilinear", "FB-MobileNetV2-P8", 160],
                ["mobilenetv2", "fast_bilinear", "FB-MobileNetV2-P2", 640],

                ["mobilenetv1", None, "MobileNetV1", None],
                ["mobilenetv1", "fast_bilinear", "FB-MobileNetV1-P4", 320],
                ["mobilenetv1", "fast_bilinear", "FB-MobileNetV1-P8", 160],
                ["mobilenetv1", "fast_bilinear", "FB-MobileNetV1-P2", 640],

                ["efficientnet", None, "EfficientNetB0", None],
                ["efficientnet", "fast_bilinear", "FB-EfficientNetB0-P4", 320],
                ["efficientnet", "fast_bilinear", "FB-EfficientNetB0-P2", 640],
                ["efficientnet", "fast_bilinear", "FB-EfficientNetB0-P8", 160],

                
                ["mobilenetv2", "fast_bilinear", "FB-MobileNetV2-P0", None],
                ["mobilenetv1", "fast_bilinear", "FB-MobileNetV1-P0", None],
                ["mobilenetv1+mobilenetv2", "bilinear", "B-MobileNets-P0", None]
    ]
    
    # # Iterate over configurations
    for backbone_model, include_pooling, model_name, channel_reducer in pretrained_configs:
        
        custom_model = CustomPretrained(base_model=backbone_model, model_name=model_name, num_classes=num_class, img_shape=input_shape, include_pooling=include_pooling, channel_reducer=channel_reducer)
        # model.build((None, 224, 224, 3))
        model = custom_model.build_model()
        model.summary()

        history = train_model(train_X, train_Y, valid_X, valid_Y, round_label, model, optimizer, num_ep, batch_size, model_name, lr_reduce, lr_reduce_factor, lr_reduce_patience, lr_min)
        save_training_log(history, round_label, model_name, num_ep)
        plot_acc(history, round_label, model_name, num_ep)
        plot_loss(history, round_label, model_name, num_ep)

def train_model(train_X, train_Y, valid_X, valid_Y, round_label, model, optimizer, num_ep, BATCH_SIZE, base_model, lr_reduce, lr_reduce_factor, lr_reduce_patience, lr_min):
    # tf.keras.backend.clear_session()
    # tf.random.set_seed(51)
    # np.random.seed(51)

    # optimizer.lr.assign(learning_rate)
        
    TRAIN_STEP_PER_EPOCH = np.ceil((train_X.shape[0]/BATCH_SIZE)-1)
    print(TRAIN_STEP_PER_EPOCH)

    train_generator, val_generator = augment_data(train_X, train_Y, valid_X, valid_Y, BATCH_SIZE)

    model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
    
    time_label = datetime.now().strftime('%y%m%d') + f'_{round_label}'

     # Define callbacks to save best model
    filepath = f'../temp/checkpoint/{base_model}_{num_ep}ep-best-{time_label}.h5'

    model_checkpoint = ModelCheckpoint(filepath, 
                                       save_best_only = True,  
                                       monitor = 'val_accuracy', 
                                       mode = 'max', 
                                       verbose = 1, 
                                       save_weights_only=True,
                                       save_freq='epoch')
      
    if lr_reduce:
        print("ReduceonPlateau is activated")
        lr_reducer = ReduceLROnPlateau(monitor='val_loss', patience=lr_reduce_patience, verbose=1, factor=lr_reduce_factor, mode='min', min_lr= lr_min)
        callback_list = [model_checkpoint, lr_reducer]
    elif lr_reduce == False or lr_reduce == None:
        "Without ReduceonPlateau"
        callback_list = [model_checkpoint]

    start = process_time()

    history = model.fit(
        # datagen.flow(train_X, train_Y, batch_size=BATCH_SIZE),
        train_generator,
        # train_X,train_Y, batch_size=BATCH_SIZE,
        callbacks=callback_list,
        epochs = num_ep,
        # validation_data=(valid_X,valid_Y),
        validation_data=val_generator,
        # shuffle = True,
        verbose = 1, 
        steps_per_epoch=TRAIN_STEP_PER_EPOCH
    )
    end = process_time()
    time = end - start
    print("Elapsed time for model in seconds: ")
    print(time)
    
    filepath = f'../temp/checkpoint/{base_model}_{num_ep}ep-last-{time_label}.h5'
    model.save_weights(filepath)

    return history

def augment_data(train_X, train_Y, valid_X, valid_Y, BATCH_SIZE):
    ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

    datagen = ImageDataGenerator(
            rotation_range=10-20,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range = 0.1-0.3, # Randomly zoom image 
            # shear_range = 0.01,
            width_shift_range=0.03,  # randomly shift images horizontally
            height_shift_range=0.03,  # randomly shift images vertically
            horizontal_flip=True, 
            vertical_flip=False,
            # rescale=1./255
            # fill_mode='nearest'
            )
    
    # Create data generators
    train_generator = datagen.flow(train_X, train_Y, batch_size=BATCH_SIZE)
    val_generator = datagen.flow(valid_X, valid_Y, batch_size=BATCH_SIZE)

    return train_generator, val_generator

def save_training_log(history, round_label, base_model, num_ep):
    # convert the history.history dict to a pandas DataFrame:     
    hist_df = pd.DataFrame(history.history) 

    hist_csv_file = f'../temp/train_log/{base_model}_{num_ep}ep-'+datetime.now().strftime('%Y%m%d')+f'_{round_label}.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

def plot_acc(history, round_label, base_model, num_ep):
    # plt.style.use("ggplot") #to add grid
    plt.figure(figsize=(10, 6))
    # plt.suptitle("Best model at epoch: " + str(np.argmin(res_adam.history["val_loss"])), y=0.92, fontsize=12)
    plt.title(f'Training and validation accuracy of {base_model} {num_ep} epoch on Fold {round_label}', y=1.06, fontsize=14)
    plt.plot(history.history["accuracy"], label="accuracy")
    plt.plot(history.history["val_accuracy"], label="val_acccuracy")
    # plt.plot(np.argmin(res_adam.history["val_loss"]), np.min(res_adam.history["val_loss"]), marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("log_acc")
    # plt.ylim(0, 100) # Set y-axis limits to 0 and 100
    plt.legend()
    plt.savefig(f'../temp/train_log/{base_model}_{num_ep}ep-acc-'+datetime.now().strftime('%y%m%d')+f'_{round_label}.png')
    plt.close()  # Close the current figure after saving

def plot_loss(history, round_label, base_model, num_ep):
    # plt.style.use("ggplot") #to add grid
    plt.figure(figsize=(10, 6))
    plt.suptitle("Best model at epoch: " + str(np.argmin(history.history["val_loss"])+1), y=0.92, fontsize=12)
    plt.title(f'Training and validation loss of {base_model} {num_ep} epoch on Fold {round_label}', y=1.06, fontsize=14)
    plt.plot(history.history["loss"], label="loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.plot(np.argmin(history.history["val_loss"]), np.min(history.history["val_loss"]), marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("log_loss")
    plt.legend()
    plt.savefig(f'../temp/train_log/{base_model}_{num_ep}ep-loss-'+datetime.now().strftime('%y%m%d')+f'_{round_label}.png')
    plt.close()  # Close the current figure after saving

def classification_report_csv(report):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:]:
        row = {}
        row_data = line.split('    ')
        if len(row_data)>9:
            row['class'] = row_data[1]
            row['f1_score'] = row_data[7]
            row['support'] = row_data[9]
        if len(row_data)>6 and len(row_data)<9:
            row['class'] = row_data[2]
            row['precision'] = row_data[3]
            row['recall'] = row_data[4]
            row['f1_score'] = row_data[5]
            row['support'] = row_data[6]
        elif len(row_data) >1 and len(row_data)<7:
            row['class'] = row_data[0]
            row['precision'] = row_data[1]
            row['recall'] = row_data[2]
            row['f1_score'] = row_data[3]
            row['support'] = row_data[4]
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    # dataframe.to_csv('classification_report.csv', index = False)
    return dataframe

# report = classification_report(y_true, y_pred)

def save_report_txt(report, filename):
    # Open the file in append mode
    text_file = open(filename, 'wt')
    text_file.write(report)
    text_file.close()

def check_predict_detail(label, predict):
    for i in range(len(predict)):
        print("Predicted: {}, Ori: {}".format(predict[i], label[i]))

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', save_as = 'title to save', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Succeed creating confusion matrix without normalization')

    # print(cm)
    plt.figure(figsize=(5, 5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.style.use('default')
    plt.title(title, y =1.1)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    # plt.show()
    plt.savefig(save_as)
    plt.close()  # Close the current figure after saving



def plot_umap(model, test_X, test_Y, filename, layer_name, x_lim=(-20, 30), y_lim=(-20,30)):
    # Extract embeddings from the model
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    embeddings = intermediate_layer_model.predict(test_X)
    
    # Use UMAP to visualize the embeddings
    umap_model = umap.UMAP(n_components=2, random_state=42)
    umap_embeddings = umap_model.fit_transform(embeddings)
    
    # Map class labels
    class_label_mapping = {0: 'Normal', 1: 'Neoplastic', 2: 'Cerebrovascular', 3: 'Degenerative', 4: 'Inflammatory'}
    class_labels = [class_label_mapping[label] for label in test_Y]

    # Create a DataFrame for the data
    df = pd.DataFrame({
        'UMAP1': umap_embeddings[:, 0],
        'UMAP2': umap_embeddings[:, 1],
        'Classes': class_labels
    })
    
    # Initialize JointGrid with correct axis limits
    g = sns.JointGrid(x='UMAP1', y='UMAP2', data=df, xlim=x_lim, ylim=y_lim)
    
    # Plot the scatterplot on the JointGrid
    sns.scatterplot(data=df, x='UMAP1', y='UMAP2', hue='Classes', alpha=0.5, ax=g.ax_joint, legend=False)
    sns.kdeplot(data=df, x='UMAP1', hue='Classes', multiple="stack", ax=g.ax_marg_x, fill=True, common_norm=False, alpha=.5, linewidth=0, legend=False)
    sns.kdeplot(data=df, y='UMAP2', hue='Classes', multiple="stack", ax=g.ax_marg_y, fill=True, common_norm=False, alpha=.5, linewidth=0, legend=False)

    # Adjust the legend to be outside the plot
    g.ax_joint.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Classes')

    # Save the plot
    g.savefig(f'../temp/test_log/umap/umap-{filename}.jpg', dpi=300)
    plt.close(g.fig)  # Close the JointGrid figure after saving