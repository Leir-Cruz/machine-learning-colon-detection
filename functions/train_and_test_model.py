import torch
import copy
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

criterion = torch.nn.CrossEntropyLoss()

def train_model(model, dataloader_train, dataloader_validation, optimizer, epochs = 10, patience = 0, scheduler = None):
    model = model.to("cuda")
    device = "cuda"
    train_losses = []
    val_losses = []
    best_loss = float("inf")
    best_model_state = None
    epochs_no_improve = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_loss_validation = 0

        for images, labels in dataloader_train:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size()[0]

        with torch.no_grad():
            model.eval() 

            for images, labels in dataloader_validation:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss_validation += loss.item() * images.size()[0]
            

        avg_train_loss = total_loss / len(dataloader_train.dataset)
        avg_val_loss = total_loss_validation / len(dataloader_validation.dataset)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        if scheduler is not None:
            scheduler.step()

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), f"{model.__class__.__name__}.pt")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        if patience > 0 and epochs_no_improve >= patience:
            print(f"\nEarly stopping ativado! Melhor Val Loss: {best_loss:.4f}")
            break
    
    model.load_state_dict(best_model_state)
    return train_losses, val_losses, model



def test_model(model, device, dataloader_test):
    model = model.to(device)
    model.eval()  # Modo de avaliação
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader_test:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Cálculo das métricas
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"Acurácia: {acc:.4f}")
    print(f"Precisão (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1-score (weighted): {f1:.4f}")
    print("Matriz de Confusão:")
    print(cm)

    return acc, precision, recall, f1, cm


def train_model_two_phases(model, dataloader_train, dataloader_validation, epochs=10, patience=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # =============================
    # CONFIGURAÇÃO DAS FASES
    # =============================
    epochs_fase1 = int(epochs * 0.3)  # 30% das épocas
    epochs_fase2 = epochs - epochs_fase1

    # Detecta backbone automaticamente
    backbone = None
    if hasattr(model, "features"):
        backbone = model.features
    elif hasattr(model, "backbone"):
        backbone = model.backbone

    train_losses = []
    val_losses = []

    best_loss = float("inf")
    best_model_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    current_epoch = 0
    total_epochs = epochs

    print(f"🚀 Treinamento em duas fases | Total de épocas: {epochs}")

    for epoch in range(total_epochs):

        # =============================
        # CONTROLE DE FASES
        # =============================
        if epoch == 0:
            print("\n❄️ FASE 1: Treinando apenas a cabeça da rede...")

            if backbone is not None:
                for param in backbone.parameters():
                    param.requires_grad = False

                params_to_train = [p for p in model.parameters() if p.requires_grad]
            else:
                # fallback: treina tudo
                params_to_train = model.parameters()

            optimizer = torch.optim.AdamW(params_to_train, lr=1e-3, weight_decay=1e-2)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_fase1)

        elif epoch == epochs_fase1:
            print("\n🔥 FASE 2: Fine-tuning completo da rede...")

            for param in model.parameters():
                param.requires_grad = True

            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-3)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_fase2)

        current_epoch += 1

        # =============================
        # TREINO
        # =============================
        model.train()
        total_loss = 0

        for images, labels in dataloader_train:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            # classificação binária com CrossEntropy → labels long
            loss = criterion(outputs, labels.long())

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)

        # =============================
        # VALIDAÇÃO
        # =============================
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for images, labels in dataloader_validation:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels.long())

                total_val_loss += loss.item() * images.size(0)

        avg_train_loss = total_loss / len(dataloader_train.dataset)
        avg_val_loss = total_val_loss / len(dataloader_validation.dataset)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        if scheduler is not None:
            scheduler.step()

        # =============================
        # MELHOR MODELO
        # =============================
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, f"{model.__class__.__name__}.pt")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        print(
            f"Epoch {current_epoch}/{total_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        )

        # =============================
        # EARLY STOPPING
        # =============================
        if patience > 0 and epochs_no_improve >= patience:
            print(f"\n⛔ Early stopping! Melhor Val Loss: {best_loss:.4f}")
            break

    model.load_state_dict(best_model_state)
    return train_losses, val_losses, model