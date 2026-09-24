import argparse
from train import train_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Siamese Network")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the training dataset folder")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    
    args = parser.parse_args()
    
    print(f"Starting training with data from: {args.data_dir}")
    train_model(args.data_dir, epochs=args.epochs, batch_size=args.batch_size)
