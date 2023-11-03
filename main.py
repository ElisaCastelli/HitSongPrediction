import argparse

def main():
    parser = argparse.ArgumentParser(
        description='Hit Song Prediction')

    parser.add_argument('--epochs', type=int, help='Number of epochs', default=5000)
    parser.add_argument('--batch_size', type=int, help='batch size', default=32)
    parser.add_argument('--number_missing_ldspk', type=int, help='number of missing loudspeakers', default=32)
    parser.add_argument('--gt_soundfield_dataset_path', type=str, help='path to dataset',
                        default='/nas/home/lcomanducci/pressure_matching_deep_learning/dataset/circular_array/gt_soundfield_train.npy')
    parser.add_argument('--learning_rate', type=float, help='Learning rate', default=0.001)



if __name__ == '__main__':
    main()