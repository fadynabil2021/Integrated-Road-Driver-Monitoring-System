import argparse
from .pipeline import enhanced_pipeline

def main():
    parser = argparse.ArgumentParser(description='Head Pose & Gaze Estimation Pipeline')
    parser.add_argument('--train_val_subjects', nargs='+', type=int, default=[1,2,3,4,5,6,7,8,9,10,12,13,14,17,18], help='List of training/validation subject numbers')
    parser.add_argument('--test_subjects', nargs='+', type=int, default=[15,19,20], help='List of test subject numbers')
    parser.add_argument('--backbone', type=str, default='mobilenet_v2', choices=['mobilenet_v2', 'efficientnet_b0'], help='Model backbone')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--skip_training', action='store_true', help='Skip training and use existing checkpoint')
    args = parser.parse_args()

    model, metrics, predictions, tracker = enhanced_pipeline(
        train_val_subjects=args.train_val_subjects,
        test_subjects=args.test_subjects,
        backbone=args.backbone,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        skip_training=args.skip_training
    )
    print('Pipeline finished.')
    if metrics:
        print('Evaluation metrics:')
        print(metrics)

if __name__ == '__main__':
    main()
