import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import collections

# Import our custom modules
from src.models.experts import Expert
from src.models.losses import LogitAdjustLoss, BalancedSoftmaxLoss
from src.metrics.calibration import TemperatureScaler

# --- CONFIGURATION (sẽ được thay thế bằng Hydra) ---
CONFIG = {
    'dataset': {
        'name': 'cifar100_lt_if100', ## CẬP NHẬT: Đổi tên cho nhất quán
        'data_root': './data',
        'splits_dir': './data/cifar100_lt_if100_splits',
        'num_classes': 100,
    },
    'experts_to_train': [
        {'name': 'ce_baseline', 'loss_type': 'ce'},
        # Bạn có thể thêm các expert khác và huấn luyện chúng với cùng công thức mạnh mẽ này
        {'name': 'logitadjust_baseline', 'loss_type': 'logitadjust'},
        {'name': 'balsoftmax_baseline', 'loss_type': 'balsoftmax'},
    ],
    'train_params': {
        'epochs': 256,         ## CẬP NHẬT: 256 epochs
        'batch_size': 128,
        'lr': 0.4,             # LR cao theo baseline
        'momentum': 0.9,
        'weight_decay': 1e-4,  # Weight decay theo baseline
        'warmup_steps': 15,    ## CẬP NHẬT: Thêm warmup
    },
    'output': {
        'checkpoints_dir': './checkpoints/experts',
        'logits_dir': './outputs/logits',
    },
    'seed': 42
}
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- HELPER FUNCTIONS ---

def get_dataloaders(config):
    # Dataloaders for training and calibration
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                            (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                            (0.2675, 0.2565, 0.2761)),
    ])

    base_train_dataset = torchvision.datasets.CIFAR100(root=config['dataset']['data_root'], train=True, transform=transform_train)
    
    ## CẬP NHẬT: Sử dụng val_lt cho việc theo dõi, thay vì val_small để nhất quán với paper
    base_val_dataset = torchvision.datasets.CIFAR100(root=config['dataset']['data_root'], train=False, transform=transform_test)

    splits_dir = Path(config['dataset']['splits_dir'])
    with open(splits_dir / 'train_indices.json', 'r') as f:
        train_indices = json.load(f)
    with open(splits_dir / 'val_lt_indices.json', 'r') as f:
        val_indices = json.load(f)

    train_dataset = Subset(base_train_dataset, train_indices)
    val_dataset = Subset(base_val_dataset, val_indices) # val_dataset giờ là val_lt

    train_loader = DataLoader(train_dataset, batch_size=config['train_params']['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['train_params']['batch_size'], shuffle=False, num_workers=4)
    
    return train_loader, val_loader

def get_loss_function(loss_type, train_loader):
    if loss_type == 'ce':
        return nn.CrossEntropyLoss()
    
    print("Calculating class counts for loss function...")
    train_targets = np.array(train_loader.dataset.dataset.targets)[train_loader.dataset.indices]
    class_counts = [count for _, count in sorted(collections.Counter(train_targets).items())]
    
    if loss_type == 'logitadjust':
        return LogitAdjustLoss(class_counts=class_counts)
    elif loss_type == 'balsoftmax':
        return BalancedSoftmaxLoss(class_counts=class_counts)
    else:
        raise ValueError(f"Loss type '{loss_type}' not supported.")

## CẬP NHẬT: Hàm export_logits được viết lại hoàn toàn để xử lý đúng các split mới
def export_logits_for_all_splits(model, config, expert_name):
    print(f"Exporting logits for expert '{expert_name}'...")
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    splits_dir = Path(config['dataset']['splits_dir'])
    output_dir = Path(config['output']['logits_dir']) / config['dataset']['name'] / expert_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Các split lấy từ TẬP TRAIN GỐC
    splits_from_train_pool = ['train', 'tuneV', 'val_small', 'calib']
    base_train_dataset = torchvision.datasets.CIFAR100(root=config['dataset']['data_root'], train=True, transform=transform)
    for name in splits_from_train_pool:
        with open(splits_dir / f"{name}_indices.json", 'r') as f:
            indices = json.load(f)
        subset = Subset(base_train_dataset, indices)
        loader = DataLoader(subset, batch_size=512, shuffle=False, num_workers=4)
        
        all_logits = []
        with torch.no_grad():
            for inputs, _ in tqdm(loader, desc=f"Exporting {name}"):
                logits = model.get_calibrated_logits(inputs.to(DEVICE))
                all_logits.append(logits.cpu())
        
        all_logits = torch.cat(all_logits)
        torch.save(all_logits.to(torch.float16), output_dir / f"{name}_logits.pt")

    # Các split lấy từ TẬP TEST GỐC (val_lt, test_lt)
    splits_from_test_pool = ['val_lt', 'test_lt']
    base_test_dataset = torchvision.datasets.CIFAR100(root=config['dataset']['data_root'], train=False, transform=transform)
    for name in splits_from_test_pool:
        with open(splits_dir / f"{name}_indices.json", 'r') as f:
            indices = json.load(f)
        subset = Subset(base_test_dataset, indices)
        loader = DataLoader(subset, batch_size=512, shuffle=False, num_workers=4)
        
        all_logits = []
        with torch.no_grad():
            for inputs, _ in tqdm(loader, desc=f"Exporting {name}"):
                logits = model.get_calibrated_logits(inputs.to(DEVICE))
                all_logits.append(logits.cpu())
            
        all_logits = torch.cat(all_logits)
        torch.save(all_logits.to(torch.float16), output_dir / f"{name}_logits.pt")


# --- MAIN TRAINING SCRIPT ---
def main():
    torch.manual_seed(CONFIG['seed'])
    np.random.seed(CONFIG['seed'])

    train_loader, val_loader = get_dataloaders(CONFIG)
    
    for expert_config in CONFIG['experts_to_train']:
        expert_name = expert_config['name']
        loss_type = expert_config['loss_type']
        print(f"\n{'='*20} Training Expert: {expert_name.upper()} {'='*20}")

        model = Expert(num_classes=CONFIG['dataset']['num_classes']).to(DEVICE)
        criterion = get_loss_function(loss_type, train_loader)
        optimizer = optim.SGD(model.parameters(), lr=CONFIG['train_params']['lr'], momentum=CONFIG['train_params']['momentum'], weight_decay=CONFIG['train_params']['weight_decay'])
        
        ## CẬP NHẬT: Sử dụng MultiStepLR thay cho Cosine
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[96, 192, 224], gamma=0.1)
        
        best_acc = 0.0
        checkpoint_dir = Path(CONFIG['output']['checkpoints_dir']) / CONFIG['dataset']['name']
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_model_path = checkpoint_dir / f"best_{expert_name}.pth"

        ## CẬP NHẬT: Thêm logic Warmup
        global_step = 0
        warmup_steps = CONFIG['train_params']['warmup_steps']
        base_lr = CONFIG['train_params']['lr']

        for epoch in range(CONFIG['train_params']['epochs']):
            model.train()
            running_loss = 0.0
            for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['train_params']['epochs']}"):
                ## CẬP NHẬT: Logic Warmup bên trong vòng lặp batch
                if global_step < warmup_steps:
                    lr_scale = (global_step + 1) / warmup_steps
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = base_lr * lr_scale
                
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                global_step += 1
            
            # scheduler.step() được gọi sau mỗi epoch
            scheduler.step()
            
            # Validation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
            
            acc = 100 * correct / total
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}: Loss={running_loss/len(train_loader):.4f}, Val Acc={acc:.2f}%, LR={current_lr:.5f}")
            
            if acc > best_acc:
                best_acc = acc
                print(f"Saving new best model to {best_model_path}")
                torch.save(model.state_dict(), best_model_path)

        # Post-Training: Calibration (sử dụng val_loader, tức là val_lt)
        print(f"\n--- Post-processing for {expert_name} ---")
        model.load_state_dict(torch.load(best_model_path))
        
        scaler = TemperatureScaler()
        optimal_temp = scaler.fit(model, val_loader, DEVICE)
        model.set_temperature(optimal_temp)
        
        final_model_path = checkpoint_dir / f"final_calibrated_{expert_name}.pth"
        torch.save(model.state_dict(), final_model_path)
        print(f"Saved final calibrated model to {final_model_path}")

        # Export Logits cho tất cả các split cần thiết
        export_logits_for_all_splits(model, CONFIG, expert_name)

if __name__ == '__main__':
    main()