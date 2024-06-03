# Imports
import pandas as pd
import numpy as np
#import tensorflow as tf
import os
import ntpath
import argparse

os.environ["KERAS_BACKEND"] = "jax" 

import keras
from keras.layers import Dense, GlobalAveragePooling2D, Flatten
from keras.models import Model
from tqdm import tqdm

def add_model_args(parser: argparse.ArgumentParser):
    """Model arguments"""

    group = parser.add_argument_group('model', 'model configuration')
    group.add_argument("--model-name", type=str)
    group.add_argument('--model-dir', type=str, help='model path')
    group.add_argument("--data-dir", type=str)
    group.add_argument("--epochs", type=int)
    group.add_argument("--learning-rate", type=float)
    group.add_argument("--batch-size", type=int)
    group.add_argument("--radchest-samples", type=int)
    group.add_argument("--split-idx",type=int)
    group.add_argument("--use-synthex", action="store_true")
    group.add_argument("--synthex-dir", type=str)
    return parser

def synthex_dir_change(path, synth_dir):
    short_path = ntpath.basename(path)
    name = os.path.splitext(short_path)[0]
    image_name = '%s.png' % (name)
    return os.path.join(synth_dir, image_name)


def sample_angles(n, center_df, angled_df, id_column):
    
    if n == 0:
        return pd.DataFrame()

    return pd.concat(
        [angled_df[angled_df[id_column] == i].sample(n=n-1, random_state=1) for i in center_df[id_column]] + [center_df]
    )

def get_n_splits(n, df):
    length = int(len(df)/(n+1))

    output_list = [df.iloc[int(length*i):int(length*(i+1))] for i in range(n-1)]
    output_list.append(df.iloc[int(length*n-1):len(df)])

    
    return output_list

# Function for creating the n-hot encoding
def get_n_hot_encoding(df, labels_to_encode):
    enc = np.zeros((len(df), len(labels_to_encode)))
    for idx, row in df.iterrows():
        for ldx, l in enumerate(labels_to_encode):
            if row[l] == 1:
                enc[idx][ldx] = 1
    return enc

# Function to change the image paths
def change_paths(df, data_directory):
    df = df.copy()


    df["ImagePath"] = df["ImagePath"].apply(lambda x: x.replace("/home/data_shares/purrlab_students/", data_directory))

    return df


# Function for running the multitask model
def PD_save_models(
        x_train, 
        y_train_pd, 
        x_val,
        y_val_pd,
        epochs, 
        lr,
        name, 
        out_dir,
        split_number,
        batch_size
    ):

    # Get the DenseNet ImageNet weights
    densenet_model = keras.applications.DenseNet121(weights='imagenet', include_top=False, input_shape=(512,512,3))

    # Specify whether to train on the loaded weights
    for layer in densenet_model.layers[:-9]:
        layer.trainable = False
    for layer in densenet_model.layers[-9:]:
        layer.trainable = True

    # Building further upon the last layer (ReLu)
    input_tensor = densenet_model.output
    x = GlobalAveragePooling2D(keepdims=True)(input_tensor)
    x = Flatten()(x)


    # Pathology detection layer
    PD = Dense(5, activation='sigmoid', name="PD_output")(x)
    
    
    model = Model(inputs=densenet_model.inputs, outputs=PD)

    adam = keras.optimizers.Adam(learning_rate = lr)

    checkpoint_filepath = out_dir + f"checkpoint.{name}_epochs_{epochs}_split_{split_number}.keras"
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_auc',
    mode='max',
    save_best_only=True)

    model.compile(
        loss={"PD_output": keras.losses.BinaryCrossentropy()}, 
        optimizer=adam, 
        metrics=['accuracy', keras.metrics.AUC(name='auc'), keras.metrics.BinaryCrossentropy(name="loss")]
    )

    model.summary()
    history = model.fit(
        x_train,
        y_train_pd,
        batch_size=128,
        epochs=epochs,  
        verbose=2,
        validation_data=(x_val, y_val_pd),
        callbacks=[model_checkpoint_callback]
    )

    model.save(out_dir + f"{name}_epochs_{epochs}_split_{split_number}.keras")
    print(history.history)
    # Get performances: train=PD, Val=TD

    return history



def main():
    # Defining the model hyperparameters

    parser = argparse.ArgumentParser()
    parser = add_model_args(parser)


    known_args, _ = parser.parse_known_args()

    name = known_args.model_name
    epochs = known_args.epochs
    learning_rate = float(known_args.learning_rate)
    data_dir = known_args.data_dir
    output_dir = known_args.model_dir
    n_radchest_samples = int(known_args.radchest_samples)
    split_idx = int(known_args.split_idx)

    #data_dir = "/dtu/p1/johlau/LabelReliability_and_PathologyDetection_in_ChestXrays/Data/"
    #output_dir = "/dtu/p1/johlau/LabelReliability_and_PathologyDetection_in_ChestXrays/ObjectDetection/models/"

    padchest_train = pd.read_csv(data_dir + 'padchest_train.csv', index_col=0)
    radchest_center = pd.read_csv(data_dir + "radchest_center.csv", index_col=0)
    radchest_angled = pd.read_csv(data_dir + "radchest_angled.csv", index_col=0)



    if known_args.use_synthex:
        print("changing radchest paths to synthex")
        synth_dir = "/dtu/p1/johlau/Thesis-Synthex/data/RAD-ChestCT-Synthex/"
        radchest_center["image_path"] = radchest_center["image_path"].apply(lambda x : synthex_dir_change(x, synth_dir))

        synth_dir = "/dtu/p1/johlau/Thesis-Synthex/data/RAD-ChestCT-Synthex-angled/"
        radchest_angled["image_path"] = radchest_angled["image_path"].apply(lambda x : synthex_dir_change(x, synth_dir))


    radchest_train = sample_angles(n_radchest_samples, radchest_center, radchest_angled, "NoteAcc_DEID")

    #train_df_all = pd.concat([padchest_train, radchest_train]).reset_index(drop=True)
    labels_to_encode = ['Effusion', 'Pneumothorax', 'Atelectasis', 'Cardiomegaly', 'Pneumonia']
    train_splits = get_n_splits(5, padchest_train.sample(frac=1, random_state=1))

    splits_ = list(train_splits)
    splits_.append(radchest_train)
    val_df = splits_.pop(split_idx).reset_index(drop=True)
    train_df = pd.concat(splits_).reset_index(drop=True)

    print(len(train_df))

    y_train = train_df[labels_to_encode].to_numpy()
    y_val= val_df[labels_to_encode].to_numpy()
    print(y_train.shape)
    print(y_val.shape)

    x_train = np.array([keras.utils.img_to_array(keras.utils.load_img(i, color_mode="grayscale").convert("RGB")) for i in tqdm(train_df["image_path"])])/255
    x_val = np.array([keras.utils.img_to_array(keras.utils.load_img(i, color_mode="grayscale").convert("RGB")) for i in tqdm(val_df["image_path"])])/255
    print(x_train.shape)
    print(x_val.shape)


    # images = np.array([keras.utils.img_to_array(keras.utils.load_img(i, color_mode="grayscale").convert("RGB")) for i in tqdm(train_df_all["image_path"])])/255
    # labels = train_df_all[labels_to_encode].to_numpy()

    # x_splits, y_splits = get_n_splits(5, images, labels)

    # train_splits = [i for i in range(5)]
    # _ = train_splits.pop(i)

    # x_train = np.concatenate([x_splits[i] for i in train_splits], axis=0)
    # y_train = np.concatenate([y_splits[i] for i in train_splits], axis=0)

    # x_val = x_splits[i]
    # y_val = y_splits[i]

    # Train model
    model_history = PD_save_models(
        x_train=x_train, 
        y_train_pd=y_train, 
        x_val=x_val,
        y_val_pd=y_val,
        epochs=epochs,
        lr=learning_rate, 
        name=name,
        out_dir=output_dir,
        split_number=split_idx,
        batch_size= int(known_args.batch_size)
    )

    # Save metrics
    filename = f"{name}_history_epochs_{epochs}_split_{split_idx}.json"
    df_history = pd.DataFrame(data=model_history.history)
    df_history.to_json(output_dir+filename)


if __name__ == "__main__":
    main()