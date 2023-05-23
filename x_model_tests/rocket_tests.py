from timeseriesAI.fastai_timeseries import *
from library.core import *

dsid = 'SmoothSubspace'
bs = 30
eps = 1e-6
n_kernels = 1000

# extract data
X_train, Y_train, X_valid, Y_valid = get_UCR_data(dsid)

# GPU ROCKET preprecessing
# normalize data 'per sample'
X_train = (X_train - X_train.mean(axis=(1, 2), keepdims=True)) / (
    X_train.std(axis=(1, 2), keepdims=True) + eps)
X_valid = (X_valid - X_valid.mean(axis=(1, 2), keepdims=True)) / (
    X_valid.std(axis=(1, 2), keepdims=True) + eps)

print(X_train.shape, X_valid.shape, Y_train.shape, Y_valid.shape)

_, features, seq_len = X_train.shape

# calculate 20k features
# When used with univariate TS,
# make sure you transform the 2d to 3d by adding unsqueeze(1).
# c_in: number of channels or features. For univariate c_in is 1.

model = ROCKET(features, seq_len, n_kernels=n_kernels, kss=[7, 9,
                                                            11]).to(device)
X_train_tfm = model(torch.tensor(X_train, device=device).float()).unsqueeze(1)
X_valid_tfm = model(torch.tensor(X_valid, device=device).float()).unsqueeze(1)

# normalize 'per feature'
f_mean = X_train_tfm.mean(dim=0, keepdims=True)
f_std = X_train_tfm.std(dim=0, keepdims=True) + eps
X_train_tfm = (X_train_tfm - f_mean) / f_std
X_valid_tfm = (X_valid_tfm - f_mean) / f_std

print(X_train_tfm.shape, X_valid_tfm.shape)

# GPU ROCKET WITH FASTAI EXAMPLE
# create databunch
data = (ItemLists('.',
                  TSList(X_train_tfm), TSList(X_valid_tfm)).label_from_lists(
                      Y_train,
                      Y_valid).databunch(bs=min(bs, len(X_train_tfm)),
                                         val_bs=min(bs * 2, len(X_valid_tfm))))


# data.show_batch()
# a Logistic Regression with 20k input features and 2 classes in this case.
def init(layer):
    if isinstance(layer, nn.Linear):
        nn.init.constant_(layer.weight.data, 0.)
        nn.init.constant_(layer.bias.data, 0.)


model = nn.Sequential(nn.Linear(n_kernels * 2, data.c))
model.apply(init)
learn = Learner(data, model, metrics=accuracy)
learn.save('stage-0')

learn.lr_find()
learn.recorder.plot()

# lr_to_use = find_appropriate_lr(learn)

learn.load('stage-0')
learn.fit_one_cycle(20, max_lr=5e-03, wd=1e2)
learn.recorder.plot_losses()
learn.recorder.plot_metrics()

# list(learn.model.parameters())

# learn_two = Learner(data, learn.model, metrics=accuracy)

# list(learn_two.model.parameters())

# try with sklearn stuff rather than fast ai
# GPU ROCKET WITH sklearn ridgeclassifiercv
X_train_tfm = X_train_tfm.reshape(X_train_tfm.shape[0], X_train_tfm.shape[2])
X_valid_tfm = X_valid_tfm.reshape(X_valid_tfm.shape[0], X_valid_tfm.shape[2])
from sklearn.linear_model import RidgeClassifierCV
ridge = RidgeClassifierCV(alphas=np.logspace(-8, 8, 17), normalize=True)
ridge.fit(X_train_tfm, Y_train)
print('alpha: {:.2E}  train: {:.5f}  valid: {:.5f}'.format(
    ridge.alpha_, ridge.score(X_train_tfm, Y_train),
    ridge.score(X_valid_tfm, Y_valid)))



# GPU ROCKET WITH sklearn LogisticRegression
eps = 1e-6
Cs = np.logspace(-5, 5, 11)
from sklearn.linear_model import LogisticRegression
best_loss = np.inf
for i, C in enumerate(Cs):
    classifier = LogisticRegression(penalty='l2', C=C, n_jobs=-1)
    classifier.fit(X_train_tfm, Y_train)
    probas = classifier.predict_proba(X_train_tfm)
    from sklearn.metrics import log_loss
    loss = log_loss(Y_train, probas)
    train_score = classifier.score(X_train_tfm, Y_train)
    val_score = classifier.score(X_valid_tfm, Y_valid)
    if loss < best_loss:
        best_eps = eps
        best_C = C
        best_loss = loss
        best_train_score = train_score
        best_val_score = val_score
    print(
        '{:2} eps: {:.2E}  C: {:.2E}  loss: {:.5f}  train_acc: {:.5f}  valid_acc: {:.5f}'
        .format(i, eps, C, loss, train_score, val_score))

# Original paper implementation
seq_len = X_train.shape[-1]
X_train = X_train[:, 0].astype(np.float64)
X_valid = X_valid[:, 0].astype(np.float64)
X_train.shape, X_valid.shape

X_train = (X_train - X_train.mean(axis=1, keepdims=True)) / (
    X_train.std(axis=1, keepdims=True) + 1e-8)
X_valid = (X_valid - X_valid.mean(axis=1, keepdims=True)) / (
    X_valid.std(axis=1, keepdims=True) + 1e-8)
X_train.mean(axis=1, keepdims=True).shape

kernels = generate_kernels(seq_len, n_kernels, kss=[7, 9, 11])

X_train_tfm = apply_kernels(X_train, kernels)
X_valid_tfm = apply_kernels(X_valid, kernels)

# ridge regression using the generated kernels
# alphas is the degree of l2 reguralization
from sklearn.linear_model import RidgeClassifierCV
classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 7), normalize=True)

classifier.fit(X_train_tfm, Y_train)
classifier.score(X_valid_tfm, Y_valid)
# 0.7279

#log reg
from sklearn.linear_model import LogisticRegression
model = sklearn.linear_model.LogisticRegression()
model.fit(X_train_tfm, Y_train)
model.score(X_valid_tfm, Y_valid)
# 0.63