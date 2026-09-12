from pathlib import Path
from copy import deepcopy
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

PROJECT_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
                    if (p / 'src/pickle/Dataset.pkl').exists())
OUT = PROJECT_ROOT / 'results/mlp_two_layer_from2020_test2026'
OUT.mkdir(parents=True, exist_ok=True)
SEED, LR = 42, 0.01
HIDDEN_LAYERS = (16, 32)
MAX_EPOCHS, PATIENCE, MIN_DELTA = 2000, 100, 1e-6
DATA_START = pd.Timestamp('2020-01-01')
TEST_START = pd.Timestamp('2026-01-01')
VAL_FRACTION = 0.05
torch.set_num_threads(1)
device = torch.device('cpu')
print('PyTorch:', torch.__version__)


dataset_path = PROJECT_ROOT / 'src/pickle/Dataset.pkl'
Dataset = pd.read_pickle(dataset_path)
X = Dataset['X'].copy()
Y = Dataset['Y'].copy()
X = X.loc[X.index >= DATA_START].copy()
Y = Y.loc[X.index].copy()
expected = ['A_500', 'Dollar Index', 'cn_10y_rate', 'us_10y_rate',
            'brent', 'Copper Futures Price', 'Gold Futures Price']
assert list(X.columns) == expected and X.index.equals(Y.index)
assert isinstance(X.index, pd.DatetimeIndex)
assert X.index.is_monotonic_increasing and X.index.is_unique
assert np.isfinite(X.to_numpy()).all() and np.isfinite(Y.to_numpy()).all()
X_train, X_test = X.loc[X.index < TEST_START].copy(), X.loc[X.index >= TEST_START].copy()
Y_train, Y_test = Y.loc[X_train.index].copy(), Y.loc[X_test.index].copy()
cut = int(len(X_train) * (1 - VAL_FRACTION))
X_fit, X_val = X_train.iloc[:cut].copy(), X_train.iloc[cut:].copy()
Y_fit, Y_val = Y_train.loc[X_fit.index], Y_train.loc[X_val.index]
VAL_START = X_val.index.min()
assert X_fit.index.max() < VAL_START <= X_val.index.min()
assert X_val.index.max() < TEST_START
assert X_fit.index.append(X_val.index).equals(X_train.index)
assert min(len(X_fit), len(X_val), len(X_test)) > 1
assert X_fit.index.max() < X_val.index.min() <= X_val.index.max() < X_test.index.min()
split_info = pd.DataFrame([
    {'数据段': name, '记录数': len(frame), '开始': frame.index.min(), '结束': frame.index.max()}
    for name, frame in [('完整训练集', X_train), ('内部拟合集', X_fit),
                        ('内部验证集', X_val), ('测试集', X_test)]])
display(split_info)


coverage = pd.DataFrame(index=X.columns)
for label, reference, future in [
    ('验证超出内部拟合范围 (%)', X_fit, X_val),
    ('测试超出内部拟合范围 (%)', X_fit, X_test),
    ('测试超出完整训练范围 (%)', X_train, X_test)]:
    coverage[label] = ((future < reference.min()) | (future > reference.max())).mean() * 100
display(coverage.round(2))

coverage.to_csv(OUT / 'coverage.csv')


scaler_x = StandardScaler().fit(X_fit)
scaler_y = StandardScaler().fit(Y_fit.to_numpy().reshape(-1, 1))
def to_tensor(values):
    return torch.tensor(np.asarray(values), dtype=torch.float32, device=device)
X_fit_tensor = to_tensor(scaler_x.transform(X_fit))
X_val_tensor = to_tensor(scaler_x.transform(X_val))
Y_fit_tensor = to_tensor(scaler_y.transform(Y_fit.to_numpy().reshape(-1, 1)))
Y_val_tensor = to_tensor(scaler_y.transform(Y_val.to_numpy().reshape(-1, 1)))
def build_mlp():
    torch.manual_seed(SEED)
    return nn.Sequential(
        nn.Linear(7, HIDDEN_LAYERS[0]), nn.ReLU(),
        nn.Linear(HIDDEN_LAYERS[0], HIDDEN_LAYERS[1]), nn.ReLU(),
        nn.Linear(HIDDEN_LAYERS[1], 1)
    ).to(device)
model = build_mlp()
loss_fn = nn.MSELoss()
print(model)
print('参数数量：', sum(p.numel() for p in model.parameters()))


optimizer = torch.optim.Adam(model.parameters(), lr=LR)
best_val_loss, best_epoch, waiting = float('inf'), 0, 0
best_state = None
history_rows = []
for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    loss = loss_fn(model(X_fit_tensor), Y_fit_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        fit_mse = loss_fn(model(X_fit_tensor), Y_fit_tensor).item()
        val_mse = loss_fn(model(X_val_tensor), Y_val_tensor).item()
    assert np.isfinite(fit_mse) and np.isfinite(val_mse)
    history_rows.append([epoch, fit_mse, val_mse])
    if val_mse < best_val_loss - MIN_DELTA:
        best_val_loss, best_epoch, waiting = val_mse, epoch, 0
        best_state = deepcopy(model.state_dict())
    else:
        waiting += 1
    if waiting >= PATIENCE:
        break
last_epoch = epoch
model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    assert np.isclose(loss_fn(model(X_val_tensor), Y_val_tensor).item(), best_val_loss)
history = pd.DataFrame(history_rows, columns=['epoch', 'fit_mse', 'val_mse'])
history.to_csv(OUT / 'selection_history.csv', index=False)
print(f'最佳轮数：{best_epoch}；实际停止轮数：{last_epoch}；'
      f'停止原因：{"连续 100 轮未明显改善" if waiting >= PATIENCE else "达到轮数上限"}')
print(f'最佳验证标准化 MSE：{best_val_loss:.6f}')
fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(history.epoch, history.fit_mse, label='Internal fit MSE')
ax.plot(history.epoch, history.val_mse, label='Validation MSE')
ax.axvline(best_epoch, color='gray', linestyle='--', label=f'Best epoch: {best_epoch}')
ax.set(xlabel='Epoch', ylabel='MSE (standardized target)', title='Select epochs using validation only')
ax.grid(alpha=0.2)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / 'selection_mse.png', dpi=140)
plt.show()


scaler_x_final = StandardScaler().fit(X_train)
scaler_y_final = StandardScaler().fit(Y_train.to_numpy().reshape(-1, 1))
X_train_tensor = to_tensor(scaler_x_final.transform(X_train))
Y_train_tensor = to_tensor(scaler_y_final.transform(Y_train.to_numpy().reshape(-1, 1)))
mlp = build_mlp()
final_optimizer = torch.optim.Adam(mlp.parameters(), lr=LR)
final_losses = []
for epoch in range(1, best_epoch + 1):
    mlp.train()
    loss = loss_fn(mlp(X_train_tensor), Y_train_tensor)
    final_optimizer.zero_grad()
    loss.backward()
    final_optimizer.step()
    mlp.eval()
    with torch.no_grad():
        final_losses.append(loss_fn(mlp(X_train_tensor), Y_train_tensor).item())
assert np.isfinite(final_losses).all()
fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(range(1, best_epoch + 1), final_losses, label='Full training MSE')
ax.set(xlabel='Epoch', ylabel='MSE (standardized target)', title='Refit on training records: 2020-2025')
ax.grid(alpha=0.2)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / 'refit_mse.png', dpi=140)
plt.show()

# 同时保存两阶段权重及各自的标准化参数；恢复时必须配套使用。
def scaler_state(scaler):
    return {'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist()}
torch.save({
    'model_state_dict': mlp.state_dict(), 'internal_best_state_dict': best_state,
    'feature_names': list(X.columns), 'hidden_layers': list(HIDDEN_LAYERS), 'seed': SEED, 'lr': LR,
    'best_epoch': best_epoch, 'stopped_epoch': last_epoch,
    'max_epochs': MAX_EPOCHS, 'patience': PATIENCE, 'min_delta': MIN_DELTA,
    'data_start': str(DATA_START.date()), 'validation_fraction': VAL_FRACTION,
    'test_start': str(TEST_START.date()), 'validation_start': str(VAL_START.date()), 'validation_end_exclusive': str(TEST_START.date()),
    'scaler_x_final': scaler_state(scaler_x_final), 'scaler_y_final': scaler_state(scaler_y_final),
    'scaler_x_internal': scaler_state(scaler_x), 'scaler_y_internal': scaler_state(scaler_y),
    'dataset_sha256': hashlib.sha256(dataset_path.read_bytes()).hexdigest()
}, OUT / 'mlp_checkpoint.pt')


def predict_price(frame):
    mlp.eval()
    with torch.no_grad():
        values = mlp(to_tensor(scaler_x_final.transform(frame))).cpu().numpy()
    return scaler_y_final.inverse_transform(values).ravel()
predictions = {}
metric_rows = []
for label, features, truth in [('Train', X_train, Y_train), ('Test', X_test, Y_test)]:
    pred = predict_price(features)
    actual = np.asarray(truth).ravel()
    residual = actual - pred
    frame = pd.DataFrame({'Actual': actual, 'Predicted': pred}, index=truth.index)
    assert np.isfinite(frame.to_numpy()).all()
    predictions[label] = frame
    frame.to_csv(OUT / f'{label.lower()}_predictions.csv')
    mse = mean_squared_error(actual, pred)
    metric_rows.append({'数据段': label, 'MSE': mse, 'RMSE': np.sqrt(mse),
                       'MAE': np.abs(residual).mean(), 'R²': r2_score(actual, pred)})
metrics = pd.DataFrame(metric_rows).set_index('数据段')
metrics.to_csv(OUT / 'metrics.csv')
display(metrics.round(4))


from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import joblib

lasso_from2020_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lasso', Lasso(max_iter=10000))
])
lasso_from2020_search = GridSearchCV(
    lasso_from2020_pipeline,
    param_grid={'lasso__alpha': np.logspace(-5, 1, 1000)},
    cv=TimeSeriesSplit(n_splits=5), scoring='neg_root_mean_squared_error',
    refit=True, return_train_score=True, n_jobs=1, error_score='raise')
lasso_from2020_search.fit(X_train, np.asarray(Y_train).ravel())
print('Lasso 最佳 alpha：', lasso_from2020_search.best_params_['lasso__alpha'])
print('五折平均验证 RMSE：', -lasso_from2020_search.best_score_)
print('最终模型迭代次数：', lasso_from2020_search.best_estimator_['lasso'].n_iter_)
joblib.dump(lasso_from2020_search.best_estimator_, OUT / 'lasso_pipeline.joblib')


lasso_cv = pd.DataFrame(lasso_from2020_search.cv_results_)
lasso_cv_alpha = lasso_cv['param_lasso__alpha'].astype(float).to_numpy()
lasso_fold_rmse = np.column_stack([
    -lasso_cv[f'split{k}_test_score'].to_numpy() for k in range(5)])
lasso_fold_train_rmse = np.column_stack([
    -lasso_cv[f'split{k}_train_score'].to_numpy() for k in range(5)])
lasso_cv_summary = pd.DataFrame({
    'alpha': lasso_cv_alpha,
    'mean_training_mse': (lasso_fold_train_rmse ** 2).mean(axis=1),
    'mean_validation_rmse': lasso_fold_rmse.mean(axis=1),
    'mean_validation_mse': (lasso_fold_rmse ** 2).mean(axis=1)})
lasso_cv_summary.to_csv(OUT / 'lasso_cv.csv', index=False)
fig, ax = plt.subplots(figsize=(13, 4))
ax.semilogx(lasso_cv_alpha, lasso_cv_summary.mean_training_mse, label='Mean training MSE (5 folds)')
ax.semilogx(lasso_cv_alpha, lasso_cv_summary.mean_validation_mse, label='Mean validation MSE (5 folds)')
ax.axvline(lasso_from2020_search.best_params_['lasso__alpha'], color='gray', linestyle='--', label='Alpha selected by RMSE')
ax.set(xlabel='Lasso alpha', ylabel='MSE (original price squared)', title='Lasso time-series cross-validation')
ax.grid(alpha=0.2)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / 'lasso_cv_mse.png', dpi=140)
plt.show()


comparison_predictions = {}
comparison_rows = []
for split, features, truth in [('Train', X_train, Y_train), ('Test', X_test, Y_test)]:
    assert predictions[split].index.equals(truth.index)
    assert np.allclose(predictions[split]['Actual'], np.asarray(truth).ravel())
    frame = predictions[split][['Actual', 'Predicted']].rename(columns={'Predicted': 'MLP'}).copy()
    frame['Lasso'] = lasso_from2020_search.predict(features)
    assert np.isfinite(frame.to_numpy()).all()
    comparison_predictions[split] = frame
    frame.to_csv(OUT / f'comparison_{split.lower()}_predictions.csv')
    for name in ['MLP', 'Lasso']:
        residual = frame.Actual - frame[name]
        mse = mean_squared_error(frame.Actual, frame[name])
        comparison_rows.append({'数据段': split, '模型': name, 'MSE': mse,
            'RMSE': np.sqrt(mse), 'MAE': np.abs(residual).mean(),
            'R²': r2_score(frame.Actual, frame[name])})
comparison_metrics = pd.DataFrame(comparison_rows).set_index(['数据段', '模型'])
display(comparison_metrics.round(4))
comparison_metrics.to_csv(OUT / 'comparison_metrics.csv')
print('测试集 RMSE 较小的模型：', comparison_metrics.loc['Test', 'RMSE'].idxmin())
print('MLP 相对 Lasso 的测试 RMSE 降幅：'
      f"{(1 - comparison_metrics.loc[('Test', 'MLP'), 'RMSE'] / comparison_metrics.loc[('Test', 'Lasso'), 'RMSE']) * 100:.2f}%（负值表示 MLP 更差）")


fig, axes = plt.subplots(2, 1, figsize=(16, 9), constrained_layout=True)
for ax, split in zip(axes, ['Train', 'Test']):
    frame = comparison_predictions[split]
    for name, color in [('Actual', 'tab:blue'), ('MLP', 'tab:orange'), ('Lasso', 'tab:green')]:
        ax.plot(frame.index, frame[name], color=color, label='MLP 7-16-32-1' if name == 'MLP' else name, linewidth=1.3, alpha=0.9)
    if split == 'Train':
        ax.axvspan(X_val.index.min(), X_val.index.max(), color='gray', alpha=0.15,
                   label='Former MLP validation (included in final training)')
    else:
        ax.axvspan(frame.index.min(), frame.index.max(), color='tab:purple', alpha=0.08)
    row_mlp = comparison_metrics.loc[(split, 'MLP')]
    row_lasso = comparison_metrics.loc[(split, 'Lasso')]
    ax.set(title=f'{split} | MLP: RMSE={row_mlp.RMSE:.4f}, R²={row_mlp["R²"]:.4f}'
           f' | Lasso: RMSE={row_lasso.RMSE:.4f}, R²={row_lasso["R²"]:.4f}',
           xlabel='Date', ylabel='Zijin price (original units)')
    ax.grid(alpha=0.2)
    ax.legend()
fig.savefig(OUT / 'mlp_lasso_predictions.png', dpi=140)
plt.show()
