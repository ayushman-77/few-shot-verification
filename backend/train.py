import torch
from torch import optim
import torch.nn.functional as F
from torchvision import transforms
from model import SiameseNetwork
from dataset import get_dataloader
from loss import ContrastiveLoss
import numpy as np
from sklearn.metrics import roc_curve, auc, f1_score, accuracy_score
from tqdm import tqdm

def train_model(data_folder, epochs=10, batch_size=32, lr=0.0005):
    """
    Training loop for the Siamese Network.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataloader = get_dataloader(data_folder, batch_size=batch_size, transform=transform)
    
    model = SiameseNetwork().to(device)
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for i, (img1, img2, label) in tqdm(enumerate(dataloader, 0), total=len(dataloader), desc=f"Epoch {epoch+1}/{epochs}"):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            
            optimizer.zero_grad()
            
            output1, output2 = model(img1, img2)
            
            loss = criterion(output1, output2, label)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss / len(dataloader):.4f}")
        
    torch.save(model.state_dict(), "siamese_model.pth")
    print("Model saved to siamese_model.pth")
    return model

def evaluate_model(model, dataloader):
    """
    Evaluates the trained model on a validation/test dataset.
    Calculates advanced metrics like ROC-AUC, F1-Score, and Optimal Threshold.
    """
    device = torch.device("cuda" if next(model.parameters()).is_cuda else "cpu")
    model.eval()
    
    total_loss = 0.0
    criterion = ContrastiveLoss()
    
    all_distances = []
    all_labels = []
    
    with torch.no_grad():
        for i, (img1, img2, label) in tqdm(enumerate(dataloader, 0), total=len(dataloader), desc="Evaluating"):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, label)
            total_loss += loss.item()
            
            distances = F.pairwise_distance(output1, output2).cpu().numpy()
            all_distances.extend(distances)
            all_labels.extend(label.cpu().numpy())
            
    avg_loss = total_loss / len(dataloader)
    
    all_labels = np.array(all_labels)
    all_distances = np.array(all_distances)
    
    similarity_scores = np.exp(-all_distances)
    true_labels = 1 - all_labels # Invert so 1 is similar, 0 is dissimilar
    
    fpr, tpr, thresholds = roc_curve(true_labels, similarity_scores)
    roc_auc = auc(fpr, tpr)
    
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    predictions = (similarity_scores >= optimal_threshold).astype(int)
    f1 = f1_score(true_labels, predictions)
    acc = accuracy_score(true_labels, predictions)
    
    print(f"--- Evaluation Metrics ---")
    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print(f"Optimal Similarity Threshold: {optimal_threshold:.4f}")
    print(f"F1-Score at Optimal Threshold: {f1:.4f}")
    print(f"Accuracy at Optimal Threshold: {acc:.4f}")
    print(f"--------------------------")
    
    return avg_loss, roc_auc, optimal_threshold
