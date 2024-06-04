
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=10)
for i, (train_index, val_index) in enumerate(skf.split(masked_index, masked_y_values)):
    print(f"{i}".center(80, "*"))

    train_mask_index = masked_index[train_index]
    val_mask_index = masked_index[val_index]
    train_mask = np.zeros_like(masked_values, dtype=bool)
    train_mask[train_mask_index] = True
    val_mask = np.zeros_like(masked_values, dtype=bool)
    val_mask[val_mask_index] = True
    print(train_mask.sum(), val_mask.sum(), test_mask.cpu().sum())

    train_mask, val_mask = torch.BoolTensor(train_mask).to(device), torch.BoolTensor(val_mask).to(device)
    model = MONet(in_dim, 16, out_dim).to(device)
    print(y_train[train_mask].shape, y_train[train_mask].sum())

    train(g, features, y_all, y_all, train_mask, val_mask, model)
    tmp_y = np.zeros_like(y_train.cpu().numpy())
    tmp_y[train_mask.cpu().numpy()] = y_train[train_mask.cpu().numpy()].cpu().numpy()
    tmp_y[val_mask.cpu().numpy()] = y_val[val_mask.cpu().numpy()].cpu().numpy()
    print("check sum")

    tmp_y_all = y_all.cpu().numpy()
    print((tmp_y != tmp_y_all).sum())

    train_mask1 = train_mask.cpu().numpy()
    val_mask1 = val_mask.cpu().numpy()
    print("train")
    print((tmp_y[train_mask1] != tmp_y_all[train_mask1]).sum())
    print((tmp_y[train_mask1] == tmp_y_all[train_mask1]).sum())
    print("val")
    print((tmp_y[val_mask1] != tmp_y_all[val_mask1]).sum())
    print((tmp_y[val_mask1] == tmp_y_all[val_mask1]).sum())


    model.load_state_dict(torch.load("./checkpoint.pt"))
    logits, test_loss, acc, auroc, aupr = evaluate(g, features, y_test, test_mask, model)
    models.append(model)

    if aupr > 0.6:
        predictions.append(logits)
    print("Test accuracy {:.4f}".format(acc))
    print("Test auroc {:.4f}".format(auroc))
    print("Test aupr {:.4f}".format(aupr))

print(predictions[0][:10])
for pred in predictions:
    print(pred.shape)

all_predictions = np.squeeze(np.array(predictions))
all_predictions = all_predictions.mean(axis=0).reshape((-1, 1))

logits = all_predictions
labels = y_test[test_mask].cpu().numpy()
print(logits.shape, labels.shape)

acc = accuracy_score(labels, np.round(logits))
auroc = roc_auc_score(labels, logits)
aupr = average_precision_score(labels, logits)

print("Test accuracy {:.4f}".format(acc))
print("Test auroc {:.4f}".format(auroc))
print("Test aupr {:.4f}".format(aupr))

