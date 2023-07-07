import matplotlib.pyplot as plt
import torch
import torchvision

from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets

from going_modular import engine
from helper_functions import plot_loss_curves
from going_modular import pred_and_plot_image
from going_modular import predictions


# Check if CUDA is available and set the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set up directory paths for train and test datasets
train_dataset = "path_to_train_dataset"
test_dataset = "path_to_test_dataset"

# Set the number of workers for data loading
NUM_WORKERS = os.cpu_count()

def create_dataloaders(train_dataset, test_dataset, transform, batch_size, num_workers=NUM_WORKERS):
    train_data = datasets.ImageFolder(train_dataset, transform=transform)
    test_data = datasets.ImageFolder(test_dataset, transform=transform)
    
    class_names = train_data.classes
    
    # Create DataLoaders
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_dataloader, test_dataloader, class_names


# Set the image size
IMG_SIZE = 224

# Define the manual transforms
manual_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# Print the manually created transforms
print(f"Manually created transforms: {manual_transforms}")

# Set the batch size
BATCH_SIZE = 32

# Create the dataloaders
train_dataloader, test_dataloader, class_names = create_dataloaders(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    transform=manual_transforms,
    batch_size=BATCH_SIZE,
    num_workers=0
)

# Get a batch of images and labels
image_batch, label_batch = next(iter(train_dataloader))
image, label = image_batch[0], label_batch[0]

# Print the image shape and label
print(image.shape, label)

# Plot the image with the corresponding label
plt.imshow(image.permute(1, 2, 0))
plt.title(class_names[label])
plt.axis(False)


class PatchEmbedding(nn.Module):
    
    def __init__(self, in_channels=3, patch_size=16, embedding_dim=768):
        super().__init__()
        
        self.patcher = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0
        )
        
        self.flatten = nn.Flatten(start_dim=2, end_dim=3)
        
    def forward(self, x):
        image_resolution = x.shape[-1]
        assert image_resolution % patch_size == 0, f"Input image resolution: {image_resolution} is not divisible by patch size: {patch_size}"
        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)
        
        return x_flattened.permute(0, 2, 1)


# Set the seed for reproducibility
def set_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

set_seeds()

# Create an instance of the PatchEmbedding module
patch_size = 16
patchify = PatchEmbedding(in_channels=3, patch_size=patch_size, embedding_dim=768)

# Print the input image size
print(f"Input image size: {image.unsqueeze(0).shape}")

# Apply patch embedding to the input image
patch_embedded_image = patchify(image.unsqueeze(0))
print(f"Output patch embedding shape: {patch_embedded_image.shape}")


class MultiheadSelfAttentionBlock(nn.Module):
    
    def __init__(self, embedding_dim=768, num_heads=12, attn_dropout=0):
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True
        )
        
    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, _ = self.multihead_attn(query=x, key=x, value=x, need_weights=False)
        return attn_output


class MLPBlock(nn.Module):
    
    def __init__(self, embedding_dim=768, mlp_size=3072, dropout=0.1):
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size, out_features=embedding_dim),
            nn.Dropout(p=dropout)
        )
            
    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x


class TransformerEncoderBlock(nn.Module):
    
    def __init__(self, embedding_dim=768, num_heads=12, mlp_size=3072, mlp_dropout=0.1, attn_dropout=0):
        super().__init__()
        
        self.msa_block = MultiheadSelfAttentionBlock(embedding_dim=embedding_dim, num_heads=num_heads, attn_dropout=attn_dropout)
        self.mlp_block = MLPBlock(embedding_dim=embedding_dim, mlp_size=mlp_size, dropout=mlp_dropout)
        
    def forward(self, x):
        x = self.msa_block(x) + x
        x = self.mlp_block(x) + x
        return x


class ViT(nn.Module):
    def __init__(
        self,
        img_size=224,
        in_channels=3,
        patch_size=16,
        num_transformer_layers=12,
        embedding_dim=768,
        mlp_size=3072,
        num_heads=12,
        attn_dropout=0,
        mlp_dropout=0.1,
        embedding_dropout=0.1,
        num_classes=1000
    ):
        super().__init__()
        
        assert img_size % patch_size == 0, f"Image size must be divisible by patch size. Image size: {img_size}, patch_size: {patch_size}"
        
        self.num_patches = (img_size * img_size) // patch_size ** 2
        
        self.class_embedding = nn.Parameter(data=torch.randn(1, 1, embedding_dim), requires_grad=True)
        self.position_embedding = nn.Parameter(data=torch.randn(1, self.num_patches + 1, embedding_dim), requires_grad=True)
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        self.patch_embedding = PatchEmbedding(in_channels=in_channels, patch_size=patch_size, embedding_dim=embedding_dim)
        self.transformer_encoder = nn.Sequential(
            *[TransformerEncoderBlock(embedding_dim=embedding_dim, num_heads=num_heads, mlp_size=mlp_size, mlp_dropout=mlp_dropout) for _ in range(num_transformer_layers)]
        )
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim, out_features=num_classes)
        )
        
        def forward(self, x):
        batch_size = x.shape[0]
        class_token = self.class_embedding.expand(batch_size, -1, -1)
        
        x = self.patch_embedding(x)
        
        x = torch.cat((class_token, x), dim=1)
        
        x = self.position_embedding + x
        
        x = self.embedding_dropout(x)
        
        x = self.transformer_encoder(x)
        
        x = self.classifier(x[:, 0])
        
        return x


# Create an instance of the ViT model
vit = ViT(num_classes=len(class_names))


# Set the optimizer and loss function
optimizer = torch.optim.Adam(params=vit.parameters(), lr=3e-3, betas=(0.9, 0.999), weight_decay=0.3)
loss_fn = torch.nn.CrossEntropyLoss()

# Set the random seed
set_seeds()

# Train the model
results = engine.train(
    model=vit,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=10,
    device=device
)

# Plot the loss curves
plot_loss_curves(results)

# Define the path to the custom image
custom_image_path = "image23_jpg"

# Make predictions and plot the image with predicted class label
pred_and_plot_image(model=vit, image_path=custom_image_path, class_names=class_names)
    
    
