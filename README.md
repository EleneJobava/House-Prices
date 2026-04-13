## Kaggle კონკურსის მიმოხილვა

Kaggle-ის კონკურსი House Prices, Iowa-ს 1460 სახლის 79 მახასიათებლის საფუძველზე გაყიდვის ფასის პროგნოზირებას. მთავარი მიზანია დავატრენინგოთ მოდელი, რომელიც პროგნოზს ვარაუდს მოგვცემს სახლის მოსალოდნელ ფასზე.

მიდგომა პრობლემის გადასაჭრელად:

ჩემს მთავარ მიდგომას წარმოადგენდა მონაცემების გამოკვლევა, განაწილებების, კორელაციების და outlier-ების პოვნა, ასევე გრაფიკების გამოყენება მონაცემთა კარგად ვიზაუალიზაციისათვის. თავდაპირველად გამოვიყენე მარტივი წრფივი მოდელები რაც რამდენიმე მიზანს ემსახურებოდა: პირველ რიგში, შევქმენი ე.წ. Baseline, ანუ საორიენტაციო წერტილი, რომელთან შედარებითაც შევაფასებდი ნებისმიერ მომდევნო რთულ ალგორითმს. წრფივი მოდელები ასევე დამეხმარა იმის დანახვაში, თუ რამდენად ეფექტური იყო ჩემს მიერ ჩატარებული Feature Engineering და რამდენად მგრძნობიარე იყო პროგნოზი რეგულარიზაციის პარამეტრების მიმართ. მას შემდეგ, რაც წრფივი მოდელებით მივაღწიე სტაბილურ შედეგს, გადავედი უფრო კომპლექსურ ალგორითმებზე.ეს მოდელები ავირჩიე იმისთვის, რომ დამეჭირა მონაცემებში არსებული არაწრფივი დამოკიდებულებები და ცვლადებს შორის რთული ინტერაქციები, რასაც სტანდარტული რეგრესია ვერ ამჩნევს. საბოლოო არჩევანი შევაჩერე CatBoost-ზე, რადგან მან აჩვენა საუკეთესო ბალანსი სიზუსტესა და განზოგადების უნარს შორის.

## რეპოზიტორიის სტრუქტურა

```
House-Prices:
    README.md -> პროექტის დეტალური აღწერა 
    model_experiment.ipynb -> ყველა ექსპერიმენტი, EDA, ტრენინგი და MLflow ლოგირება.
    model_inference.ipynb -> საუკეთესო მოდელის გამოყენება და პროგნოზების გენერაცია.
    feature_names.json -> Model Registry-სთვის შენახული feature სახელები
    skewed_features.json -> log1p ტრანსფორმაციისთვის შენახული skewed feature-ების სია
    train-ის სვეტების სია alignment-ისთვის
    lot_frontage_medians.json & garage_area_medians.json: ტრენინგზე დათვლილი მედიანები ინფერენსისთვის.

```
---

## Preprocessing & Feature Engineering

### Outliers და Log-Transformation

მონაცემების გასუფთავების ეტაპზე გამოვიყენე z-score მეთოდი (threshold=3), რამაც საშუალება მომცა ავტომატურად ამომეღო ის სტრიქონები, სადაც ნებისმიერი რიცხვითი სვეტი ექსტრემალურად გადახრილი იყო. ამ ნაბიჯმა RMSE დაახლოებით 27%-ით შეამცირა (25,820-მდე), რაც კიდევ ერთხელ ადასტურებს წრფივი მოდელების მგრძნობელობას აუთლაიერების მიმართ. ამის შემდეგ მივმართე np.log1p ტრანსფორმაციას SalePrice-ისთვის. ამან მონაცემთა განაწილება ნორმალურთან მიაახლოვა, რის შედეგადაც ცდომილება კიდევ 10%-ით შემცირდა

მოდელის სიზუსტის გაზრდის მიზნით, განვახორციელე მონაცემთა შევსების კომპლექსური სტრატეგია. იმ სვეტებში, სადაც NaN ნიშნავდა მახასიათებლის არარსებობას (მაგ: აუზი ან სარდაფი), გამოვიყენე "None" შევსება, ხოლო კატეგორიული სვეტებისთვის - Mode fill. განსაკუთრებული ყურადღება დავუთმე LotFrontage-ს, რომელიც მეზობელი უბნების (Neighborhood) მედიანით შევავსე. გარდა ამისა, შევქმენი ახალი, აგრეგირებული ფუნქციები: TotalArea, TotalBaths და HouseAge. ამ ცვლილებებმა მოდელს უფრო პირდაპირი და მნიშვნელოვანი ინფორმაცია მიაწოდა, რამაც RMSE 19,479-მდე დაიყვანა (16%-იანი გაუმჯობესება). თვეების დასაკოდირებლად გამოვიყენე Sin/Cos ციკლური ენკოდინგი, რათა მოდელს აღექვა კავშირი დეკემბერსა და იანვარს შორის.

### SalePrice-ის განაწილება

```python
print("Skewness: %f" % y_full.skew())   # → 1.882876
print("Kurtosis: %f" % y_full.kurt())   # → 6.536282
```

SalePrice-ს right-skewed განაწილებაა ეს ნიშნავს, რომ ცოტა ძალიან ძვირი სახლი ასწევს საშუალო ფასს. ეს პრობლემაა წრფივი მოდელებისთვის, ამიტომ გამოვიყენე np.log1p(SalePrice) ტრანსფორმაცია და target გავხადე სიმეტრიული.

### Missing Values

რამდენიმე სვეტს ჰქონდა ბევრი NaN, მათ შორის:
- `PoolQC` — 99.5% NaN (pool-ი თითქმის არ არის)
- `Alley` — 93.8% NaN
- `MiscFeature` — 96.3% NaN
- `LotFrontage` — 17.7% NaN

gრაფიკი გვიჩვენებდა, რომ NaN ამ სვეტებში სემანტიკური მნიშვნელობა ჰქონდა, ამიტომ ეს NaN-ები "None"-ით შევავსე.

### კატეგორიული ცვლადების კოდირება

#### Pipeline v1 — LabelEncoder (მცდარი მიდგომა)

```python
le = LabelEncoder()
for col in categorical_cols:
    train[col] = le.fit_transform(train[col])
```

**პრობლემა:** `Neighborhood` სვეტი მნიშვნელობებს ხდებდა 0, 1, 2, 3... წრფივი მოდელი ფიქრობდა, რომ "CollgCr" (=5) ორჯერ "ღირებულია" ვიდრე "Blueste" (=2). ეს მათემატიკურად არასწორია ნომინალური ცვლადებისთვის.

#### Pipeline v2 — Ordinal + One-Hot Encoding (სწორი მიდგომა)

**ხარისხობრივი სვეტები (Ordinal Encoding):**
```python
quality_map = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
ordinal_cols = ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC",
                "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond", "PoolQC"]
```

ეს სვეტები ბუნებრივი თანმიმდევრობით (**Poor → Fair → Average → Good → Excellent**) ფარავს, სათვალავი მნიშვნელობა შესაბამისია.

**ნომინალური სვეტები (One-Hot Encoding):**
```python
nominal_cols = [c for c in train.select_dtypes(include="object").columns]
train = pd.get_dummies(train, columns=nominal_cols)
train, test = train.align(test, join="left", axis=1, fill_value=0)
```

`Neighborhood`, `MSZoning`, `RoofStyle` და სხვ. — მათ თანმიმდევრობა არ გააჩნიათ, ამიტომ One-Hot კოდირება ქმნის ბინარულ სვეტებს თითოეული მნიშვნელობისთვის. `train.align(test)` უზრუნველყოფს, რომ Train და Test სეტებს ერთი და იგივე სვეტები ჰქონდეთ (inference-ში კრიტიკულია).

### NaN მნიშვნელობების დამუშავება

| სტრატეგია | სვეტები | რატომ |
|-----------|---------|-------|
| **"None" fill** | PoolQC, Alley, FireplaceQu, BsmtQual, GarageType, Fence, MiscFeature და სხვ. | NaN ნიშნავს "feature არ არსებობს" — ეს ინფორმაციაა, 0 კი არ ყოფილა |
| **Mode fill** | MSZoning, KitchenQual, Electrical, Functional, MasVnrType და სხვ. | მცირე რაოდენობის გამოტოვება; ყველაზე გავრცელებული მნიშვნელობა |
| **Median fill by Neighborhood** | LotFrontage, GarageArea | ერთ უბანში სახლებს მსგავსი ლოტი და გარაჟი აქვთ |
| **Zero fill** | GarageYrBlt, MasVnrArea, BsmtFinSF1, BsmtFinSF2 და სხვ. | სახლებს, რომლებსაც გარაჟი/სარდაფი არ აქვთ, 0 აქვთ |

**Neighborhood-ის მიხედვით Median fill:**
```python
lot_medians = train.groupby("Neighborhood")["LotFrontage"].median().to_dict()
train["LotFrontage"] = train.apply(
    lambda r: lot_medians.get(r["Neighborhood"], overall_median)
    if pd.isna(r["LotFrontage"]) else r["LotFrontage"], axis=1
)
```
ეს გვიხსნის "გლობალური median"-ის გამოყენების პრობლემას — NoRidge უბნის სახლს NoRidge-ის lot frontage median ეყენება, არა მთელი dataset-ის.

### ახალი Feature-ები

| Feature | ფორმულა | მნიშვნელობა |
|---------|---------|-------------|
| `TotalArea` | GrLivArea + TotalBsmtSF | მთლიანი ფართი — ძლიერი predictor |
| `TotalBaths` | FullBath + BsmtFullBath + 0.5×(HalfBath + BsmtHalfBath) | სანტ. კვანძების რაოდენობა |
| `TotalPorch` | OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch | სულ ვერანდის ფართი |
| `HouseAge` | YrSold − YearBuilt | ახლად აშენებული სახლები უფრო ძვირია |
| `RemodAge` | YrSold − YearRemodAdd | ბოლო რემონტიდან გასული დრო |
| `OverallScore` | OverallQual × OverallCond | ხარისხი × მდგომარეობა |
| `LivArea_Qual` | GrLivArea × OverallQual | ფართი + ხარისხი ინტერაქცია |
| `LivArea_Age` | GrLivArea / (HouseAge + 1) | ახლი სახლი + დიდი ფართი = ძვირი |
| `Qual_TotalArea` | OverallQual × TotalArea | ხარისხი × ფართი ინტერაქცია |
| `Pool` | (PoolArea > 0).astype(int) | Pool-ის ყოლა/არყოლა |
| `2ndFloor` | (2ndFlrSF > 0).astype(int) | მე-2 სართული |
| `Garage` | (GarageCars > 0).astype(int) | გარაჟი |
| `Bsmt` | (TotalBsmtSF > 0).astype(int) | სარდაფი |
| `Fireplace` | (Fireplaces > 0).astype(int) | ბუხარი |
| `Porch` | (TotalPorch > 0).astype(int) | ვერანდა |
| `MoSoldsin` | sin(2π × MoSold / 12) | თვის ციკლური კოდირება — იანვარი ≈ დეკემბერი |
| `MoSoldcos` | cos(2π × MoSold / 12) | თვის ციკლური კოდირება |

**MoSold ციკლური კოდირება:** რეგულარული LabelEncoder-ი ან OHE თვისთვის ვერ გამოხატავდა, რომ თვე 12 (დეკემბერი) მახლობლობაშია თვე 1 (იანვართან). sin/cos transform-ი ამ ციკლურ ბუნებას ინახავს.

### Skewness კორექცია

```python
skewed_feats = train[numeric_cols].apply(lambda x: x.skew())
skewed_feats = skewed_feats[skewed_feats.abs() > 0.75].index.tolist()
for feat in skewed_feats:
    train[feat] = np.log1p(train[feat].clip(lower=0))
```

Feature-ები, რომელთა skewness > 0.75, `log1p` ტრანსფორმაციას გაივლიან. ეს:
- ნორმალურ განაწილებასთან მიახლოებს მათ
- ამცირებს outlier-ების გავლენას
- გაუმჯობესებს წრფივი მოდელების შესრულებას
- inference-ში `skewed_features.json` ფაილიდან ჩაიტვირთება იგივე სია

---

## Cleaning (Outlier Removal)

Outlier-ები მნიშვნელოვანია, რადგან ისინი **გადამეტებულ გავლენას** ახდენენ მოდელზე, განსაკუთრებით წრფივ მოდელებზე. ვიზუალიზაცია (scatter plot, boxplot) გვეხმარება მათ ამოცნობაში.

**რატომ პლოტ გვჭირდება Outlier-ების ნახვისთვის:**
- boxplot-ი გვიჩვენებს IQR-დან გასულ წერტილებს
- scatter plot (GrLivArea vs SalePrice) გვიჩვენებს, სად ზის "ყალბი" outlier — 5000+ კვ.ფუტი, მაგრამ $200K-ზე ნაკლები

### მიდგომა 1: Z-Score (Pipeline v1)

```python
from scipy import stats
z_scores = stats.zscore(train[numeric_cols])
mask = (np.abs(z_scores) < 3).all(axis=1)
train = train[mask]
```

**Z-Score ფორმულა:** z = (x − μ) / σ

Z-Score > 3 ნიშნავს, რომ მნიშვნელობა 3 სტანდარტული გადახრით შორს არის საშუალოდან. ეს ამოიღებდა ყველა სტრიქონს, გ სადაც **ნებისმიერ** სვეტში z > 3.

**შედეგი:** Dataset 1460-დან ~960 სტრიქონამდე შემცირდა (~34% დაიკარგა). ეს ზედმეტად აგრესიული მიდგომა იყო — ბევრი ვალიდური მონაცემი ამოიღო. XGBoost-ს განსაკუთრებით ეზარება მონაცემების სიმცირე, ამიტომ ეს v1-ში XGBoost-ის ცუდი შესრულების ერთ-ერთი მიზეზი გახდა.

**რატომ დავამატე Z-Score Pipeline v1-ში:** Linear Regression ძალიან მგრძნობიარეა outlier-ების მიმართ (RMSE 35,312 → 25,820 Outlier Removal-ის შემდეგ). მაგრამ მივხვდი, რომ Z-Score ზედმეტ მონაცემს წყვეტს. ამიტომ v2-ში targeted მიდგომაზე გადავედი.

### მიდგომა 2: Targeted Removal (Pipeline v2)

```python
# Kaggle-ში ცნობილი anomalies ამ dataset-ში
train = train[~((train["GrLivArea"] > 4000) & (train["SalePrice"] < 300000))]
train = train[train["LotArea"] < 100000]
train = train[train["SalePrice"] < 700000]
```

**შედეგი:** მხოლოდ ~3% მოინიდა. Dataset თითქმის სრული დარჩა.

ეს outlier-ები ამ dataset-ში **documented anomalies** არიან — სახლები, რომლებიც ყიდვა-გაყიდვის არა-ბაზარი პირობებში გაიყიდა.

---

## Feature Selection

Feature Selection-ი ამცირებს overfitting-ის რისკს, გაუმჯობესებს GreenGeneralization-ს და ამცირებს ტრენინგის დროს.

### მიდგომა 1: SelectKBest (f_regression, k=30)

```python
from sklearn.feature_selection import SelectKBest, f_regression
selector = SelectKBest(f_regression, k=30)
X_selected = selector.fit_transform(X_train, y_train)
```

`f_regression` feature-ებს აფასებს target-თან **წრფივი კორელაციის** (F-statistics) მიხედვით:
- F-statistic ზომავს, to რამდენად კარგად ხსნის feature მარტო target-ს
- ეს **ინდივიდუალური** შეფასებაა — feature-ებს შორის **ინტერაქცია** არ განიხილება

**შედეგი:** Linear Regression RMSE **გაუარესდა** (23,190 → 19,784). მიზეზი:
- k=30 ძალიან ცოტა feature-ია; ბევრი სასარგებლო ინფორმაცია გამოიგდო
- Ridge და Lasso-ს შემთხვევაში regularization-ი **თავადვე ასრულებს** feature selection-ს — ზედმეტი იყო

### მიდგომა 2: SelectFromModel (Random Forest)

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
rf_selector = RandomForestRegressor(n_estimators=100, random_state=42)
rf_selector.fit(X_train, y_train)
selector = SelectFromModel(rf_selector, prefit=True)
X_train_selected = selector.transform(X_train)
```

Random Forest feature_importances_-ი **არაწრფივ კავშირებსაც** ითვალისწინებს → XGBoost-ისთვის ეს უფრო შესაბამისი feature-ებს გამოარჩევს.

---

## Training — ექსპერიმენტები

### Pipeline v1-ით (LabelEncoder + Z-Score Outlier Removal)

#### ექსპერიმენტი 1: Raw Data (Linear Regression)

მხოლოდ raw data-ზე (NaN-ების გარეშე დამუშავებით), LabelEncoder კოდირებით.

- **Train RMSE: ~35,312 | R²: 0.837**

**ანალიზი:** ეს "baseline"-ი არის. RMSE მაღალია, რადგან outlier-ები, NaN-ები და LabelEncoder-ის ყალბი რანჟირება მოდელს ართმევს.

---

#### ექსპერიმენტი 2: Outlier Removal (Z-Score)

Z-Score > 3-ით outlier-ების moxsna.

- **Train RMSE: 25,820 | R²: 0.873**
- **გაუმჯობესება:** ~27% RMSE შემცირება

**ანალიზი:** Linear Regression **ძალიან მგრძნობიარეა** outlier-ების მიმართ, რადგან ის ამ ექსტრემალური მნიშვნელობებს "მიჰყვება" (least squares optimization). Outlier-ის ამოღება 27%-ის გასაომჯობესებელი ეფექტი კარგი დემონსტრაციაა ამ ფენომენისა.

---

#### ექსპერიმენტი 3: Log Transform of Target

`np.log1p(SalePrice)` ტრანსფორმაცია.

- **Train RMSE: 23,190 | R²: 0.879**
- **გაუმჯობესება:** კიდევ ~10% შემცირება

**ანალიზი:** Log transform-ი **ასიმეტრიულ** target-ს სიმეტრიულს ხდის → წრფივი მოდელი უკეთ "ხედავს" ფასების განაწილებას.

---

#### ექსპერიმენტი 4: Underfit — Linear Regression (3 Feature)

მხოლოდ 3 feature: `GrLivArea`, `OverallQual`, `YearBuilt`.

- **Train RMSE: 28,954 | Test RMSE: 28,020 | R²: 0.827**

**ანალიზი — Underfitting:** Train და Test RMSE თითქმის **იდენტურია** — ეს **underfitting-ის კლასიკური ნიშანი** (High Bias). მოდელი ზედმეტად მარტივია, 3 feature სახლის ფასს ვერ ხსნის სათანადოდ. ამ შემთხვევაში regularization-ი არ დაეხმარება — მოდელს მეტი ინფორმაცია სჭირდება.

---

#### ექსპერიმენტი 5: NaN Fill + Feature Engineering

TotalArea, TotalBaths, TotalPorch, Binary flags, Cyclical MoSold.

- **Train RMSE: ~19,479 | R²: 0.910**
- **გაუმჯობესება:** ~16%

**ანალიზი:** Feature Engineering-მა ახალი ინფორმაცია შემატა — მოდელმა "გაიგო" TotalArea (ფართი), TotalBaths (სველი კვანძები) და ა.შ.

---

#### ექსპერიმენტი 6: Underfit — Ridge (alpha=10000)

ძალიან მაღალი regularization.

- **Train RMSE: 23,461 | Test RMSE: 23,184 | R²: 0.855**

**ანალიზი — Underfitting:** alpha=10000 იმდენად "ასჯარიმებს" კოეფიციენტებს, რომ ისინი თითქმის ნოლამდე ეცემა → მოდელი "ვერ სწავლობს". Train ≈ Test RMSE — High Bias.

---

#### ექსპერიმენტი 7: Overfit Attempt — Ridge (alpha=0.0001)

ძალიან დაბალი regularization.

- **Train RMSE: 17,851 | Test RMSE: 17,016 | R²: 0.921**

**ანალიზი:** Ridge-ი დიზაინით **overfitting-ს ეწინააღმდეგება** — კოეფიციენტების L2 penalty Norm ამცირებს მათ. მაშინაც კი, როცა alpha ძალიან პატარაა, ეს პენალტი მუშაობს. Train ≈ Test — **კარგი ბალანსი**.

---

#### ექსპერიმენტი 8: Ridge — GridSearchCV

```python
param_grid = {"alpha": [0.001, 0.01, 0.1, 1, 10, 50, 100, 500, 1000]}
grid = GridSearchCV(Ridge(), param_grid, cv=5, scoring="neg_mean_squared_error")
```

საუკეთესო: **alpha=50**

- **Train RMSE: 18,176 | Test RMSE: 17,208 | R²: 0.919**

**ანალიზი:** GridSearchCV cross-validation-ით პოულობს "ოქროს საშუალოს" alpha-სთვის. Train-Test gap მიახლოვებულია — **კარგი ბალანსი**, არ არის overfit/underfit.

---

#### ექსპერიმენტი 9: Underfit — Lasso (alpha=10)

ძალიან მაღალი regularization.

- **Train RMSE: 33,403 | Test RMSE: 34,843 | R²: 0.717**

**ანალიზი — Underfitting:** Lasso-ს L1 regularization feature-ებს **ნოლამდე** (ზუსტად 0) დაყავს. alpha=10 ძალიან დიდია — თითქმის ყველა კოეფიციენტი ნულია → High Bias, ცუდი მოდელი. Ridge-ისგან განსხვავებით, Lasso გაცილებით მცირე alpha-ზეც კი იწვევს underfitting-ს.

---

#### ექსპერიმენტი 10: Lasso — GridSearchCV

საუკეთესო alpha CrossValidation-ით.

- **Train RMSE: 18,240 | Test RMSE: 17,207 | R²: 0.922**

**ანალიზი:** Ridge-ის მსგავსი შედეგი. Lasso-ს feature selection-ი (ნოლამდე დაყვანა) ამ dataset-ზე მნიშვნელოვანი უპირატესობა არ გამოდგა.

---

#### ექსპერიმენტი 11: Feature Selection — SelectKBest (k=30)

- **Linear Regression Test RMSE: 19,784**

**ანალიზი:** **გაუარესდა** baseline-თან შედარებით. SelectKBest ძალიან ბევრ სასარგებლო feature-ს კარგავდა. Ridge/Lasso-ს regularization-ი უფრო კარგ "feature selection"-ს ასრულებდა.

---

#### ექსპერიმენტი 12: Overfit — Decision Tree (max_depth=None)

```python
dt = DecisionTreeRegressor(max_depth=None, random_state=42)
```

- **Train RMSE: ~0 | Test RMSE: 34,557 | R²: 0.689**

**ანალიზი — Extreme Overfitting:** `max_depth=None` ნიშნავს, რომ ხე გაიქანებს სანამ ყველა leaf node **წმინდა არ გახდება** (ანუ ერთი მნიშვნელობა). Train RMSE ≈ 0 — **ზეიდეალური fit** სატრენინგო მონაცემებზე. Test RMSE 34,557 — **ძალიან ცუდი ზოგადიზება**. ეს High Variance / Pure Overfitting-ი.

---

#### ექსპერიმენტი 13: Underfit — Decision Tree (max_depth=2)

- **Train RMSE: 38,587 | Test RMSE: 39,315 | R²: 0.613**

**ანალიზი — Underfitting:** მხოლოდ 4 ფოთოლი — ძალიან მარტივი სტრუქტურა. ვერ ხსნის ფასის სირთულეს. ეს High Bias.

---

#### ექსპერიმენტი 14: Decision Tree Tuned (max_depth=5)

- **Train RMSE: 22,412 | Test RMSE: 26,831 | R²: 0.791**

**ანალიზი:** max_depth=5 "ოქროს შუალედია" — Trail>Test gap-ი გონივრულია. მაგრამ Ridge-თან შედარებით კიდევ სუსტი — Decision Tree-ი ინდივიდუალურად Ridge-ს ვერ ასჯობნის.

---

#### ექსპერიმენტი 15: Overfit — Random Forest (max_depth=None)

- **Train RMSE: 9,341 | Test RMSE: 19,113 | R²: 0.877**

**ანალიზი — Overfitting:** Bagging (ბევრი ხე + საშუალო) ამcirobs variance-ს Decision Tree-სთან შედარებით, მაგრამ overfitting მაინც შესამჩნევია — Train RMSE ≪ Test RMSE.

---

#### ექსპერიმენტი 16: Underfit — Random Forest (max_depth=2)

- **Train RMSE: 31,883 | Test RMSE: 29,762 | R²: 0.735**

**ანალიზი — Underfitting:** max_depth=2 ძალიან ზოგადი მოდელია Random Forest-ისთვისაც. High Bias.

---

#### ექსპერიმენტი 17: Random Forest — GridSearchCV

```python
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5]
}
```

- **Train RMSE: 11,206 | Test RMSE: 19,245 | R²: 0.877**

**ანალიზი:** Overfitting მაინც შეინიშნება. Random Forest ამ dataset-ზე Ridge/Lasso-ს ვერ ჯობნის — მცირე dataset-ზე Bagging-ი ყოველთვის საუკეთესო არ არის.

---

#### ექსპერიმენტი 18: Extreme Overfit — XGBoost (Pipeline v1)

```python
xgb_model = XGBRegressor(n_estimators=1000, max_depth=9, learning_rate=0.1)
```

- **Train RMSE: 154 (!!) | Test RMSE: 22,801 | R²: 0.842**

**ანალიზი — Extreme Overfitting:** ეს v1-ის XGBoost-ი **ყველაზე ცუდი ექსპერიმენტია** — Train RMSE 154, Test RMSE 22,801. Train/Test ratio ≈ 148! მიზეზები:
1. **LabelEncoder-ის ყალბი რანჟირება** — XGBoost split-ებს ყალბ threshold-ებზე ამყარებს
2. **Z-Score Outlier Removal** — dataset 960 სტრიქონამდე შემცირდა; XGBoost-ს მეტი მონაცემი სჭირდება
3. **Regularization-ის ნაკლებობა** — `reg_alpha`, `reg_lambda` არ იყო კარგად დაყენებული

---

#### ექსპერიმენტი 19: Underfit — XGBoost (Pipeline v1)

```python
xgb_model = XGBRegressor(n_estimators=10, max_depth=2, learning_rate=0.01)
```

- **Train RMSE: 63,531 | Test RMSE: 62,032 | R²: 0.112**

**ანალიზი — Extreme Underfitting:** `n_estimators=10`, `max_depth=2`, `learning_rate=0.01` — სამივე Hyperparameter ყველაზე მცირე მნიშვნელობაზეა. მოდელს **არ ჰქონდა საკმარისი სიმძლავრე** სწავლისთვის. R²=0.112 ნიშნავს, რომ baseline (average ფასი) ამ მოდელს ჯობია.

---

#### ექსპერიმენტი 20: CatBoost Baseline (Pipeline v1)

```python
catboost = CatBoostRegressor(
    iterations=1000, depth=6, learning_rate=0.05,
    l2_leaf_reg=3, verbose=0
)
```

- **Train RMSE: 6,880 | Test RMSE: 16,656 | R²: 0.912**

**ანალიზი:** CatBoost-ს **ჩაშენებული categorical encoding** აქვს — LabelEncoder-ის პრობლემა ნაკლებად ატყობდა. Test RMSE 16,656 — **საუკეთესო შედეგი v1-ში**. თუმცა Train/Test gap-ი (6,880 vs 16,656) overfitting-ზე მიუთითებს. Pipeline v2-ით გაუმჯობესდება.

---

### Pipeline v2-ით (One-Hot Encoding + Targeted Outlier Removal + Skew Correction + HouseAge)

Pipeline v2-ის ძირითადი ცვლილებები გამოდგა:
- **One-Hot Encoding** ამოხსნა ყალბი რანჟირება → XGBoost/CatBoost "ხედავს" სწორ feature-ებს
- **Targeted Outlier Removal** 97% მონაცემები შეინარჩუნა → ხე-ბაზირებულ მოდელებს მეტი მონაცემი
- **HouseAge**, **OverallScore** → ახალი ძლიერი predictors
- **Skew Correction** → feature-ების განაწილება გაუმჯობესდა

**CatBoost v2 (საბოლოო მოდელი):**

```python
catboost = CatBoostRegressor(
    iterations=2000, depth=6, learning_rate=0.03,
    l2_leaf_reg=3, subsample=0.8, colsample_bylevel=0.8,
    verbose=0
)
```

- **Train RMSE: ~5,200 | Test RMSE: ~14,100 | R²: ~0.935**

---

## რატომ სჯობდა CatBoost?

CatBoost **საუკეთესო მოდელია** ამ dataset-ზე რამდენიმე მიზეზის გამო:

1. **Gradient Boosting** — ID (iterative) ეტაპებზე ასწავლიდა შეცდომებს
2. **Ordered Boosting** — overfitting-ს ეწინააღმდეგება Pipeline v1-ის შედეგებთან შედარებით
3. **Regularization-ი** — `l2_leaf_reg`, `subsample`, `colsample_bylevel` overfitting-ს ამcirobs
4. **Pipeline v2-ის გაუმჯობესებები** — better encoding, more data, informative features

### Pipeline v1-ზე XGBoost-ს რატომ სჯობდა Ridge/Lasso?

1. **მცირე dataset (960 სტრიქონი Z-Score შემდეგ):** XGBoost-ს მეტი მონაცემი სჭირდება
2. **LabelEncoder-ის ყალბი რანჟირება:** Ridge regularization-ი ამ ყალბ კოეფიციენტებს ამcirobs, XGBoost split-ებს ყალბ threshold-ებზე ამყარებდა
3. **SelectKBest:** წრფივი feature შერჩევა XGBoost-ისთვის არ განხდა

---

## MLflow Tracking

### ექსპერიმენტების ბმული

🔗 https://dagshub.com/ejoba22/House-Prices.mlflow

### MLflow-ს კავშირი

```python
import os
import mlflow

os.environ['MLFLOW_TRACKING_USERNAME'] = 'ejoba22'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '...'

mlflow.set_tracking_uri("https://dagshub.com/ejoba22/House-Prices.mlflow")
mlflow.set_experiment("house-prices")
```

### ჩაწერილი მეტრიკები

ყველა ექსპერიმენტი ლოგ:

| მეტრიკა/პარამეტრი | აღწერა |
|-------------------|--------|
| `train_rmse` | სატრენინგო RMSE (expm1-ით უკუტრანსფორმირებული) |
| `test_rmse` | სატესტო RMSE (Validation set) |
| `r2` | R-squared score Test set-ზე |
| `model_type` | მოდელის ტიპი (Ridge, XGBoost, CatBoost...) |
| `feature_selection` | Feature selection მეთოდი |
| `outlier_method` | Outlier removal მეთოდი |
| `encoding` | Encoding მეთოდი (LabelEncoder/OHE) |
| `best_params` | GridSearchCV-ის საუკეთესო პარამეტრები |
| `pipeline_version` | v1 ან v2 |

### MLflow Run Logging

```python
with mlflow.start_run(run_name="CatBoost_v2"):
    mlflow.log_param("model_type", "catboost")
    mlflow.log_param("pipeline_version", "v2")
    mlflow.log_param("encoding", "OHE")
    mlflow.log_metric("train_rmse", train_rmse)
    mlflow.log_metric("test_rmse", test_rmse)
    mlflow.log_metric("r2", r2)
    mlflow.catboost.log_model(model, "catboost_model",
                              registered_model_name="catboost_baseline")
```

### Model Registry

საუკეთესო მოდელი (CatBoost) **MLflow Model Registry**-ში არის რეგისტრირებული:
- **Model Name:** `catboost_baseline`
- **Version:** 3

---

## Model Inference (model_inference.ipynb)

### მოდელის ჩამოტვირთვა Model Registry-დან

```python
model = mlflow.catboost.load_model("models:/catboost_baseline/3")
print("Model loaded successfully!")
print(type(model))  # → <class 'catboost.core.CatBoostRegressor'>
```

### Test Set-ზე პროგნოზი

```python
# Predict in log-space
predictions_log = model.predict(test)

# Reverse log1p transform
predictions_final = np.expm1(predictions_log)

# Clip unreasonably low values
predictions_final = np.clip(predictions_final, a_min=10000, a_max=None)

submission = pd.DataFrame({'Id': test_ids, 'SalePrice': predictions_final})
submission.to_csv('submission.csv', index=False)
```

### Submission შედეგები

```
count    1459.000000
mean   178219.977738
std     77475.263010
min     43588.059053
25%    127372.374128
50%    155855.812462
75%    207268.016347
max    557067.462481
```

---

## შეჯამება — ყველა მოდელის შედარება (Pipeline v1)

| მოდელი | Train RMSE | Test RMSE | R² | სტატუსი |
|--------|-----------|-----------|-----|---------|
| CatBoost | 6,880 | **16,656** | 0.912 | Slight overfitting |
| Ridge (α=0.0001) | 17,851 | **17,016** | 0.921 | კარგი ბალანსი |
| Lasso (GridSearchCV) | 18,240 | **17,207** | 0.922 | კარგი ბალანსი |
| Ridge (GridSearchCV) | 18,176 | **17,208** | 0.919 | კარგი ბალანსი |
| LR + SelectKBest | 20,498 | 19,784 | 0.886 | კარგი |
| RF Tuned (GridSearchCV) | 11,206 | 19,245 | 0.877 | Overfitting |
| RF Overfit (max_depth=None) | 9,341 | 19,113 | 0.877 | Overfitting |
| DT Tuned (max_depth=5) | 22,412 | 26,831 | 0.791 | კარგი |
| XGBoost Overfit | **154** | 22,801 | 0.842 | **Extreme Overfit** |
| LR Underfit (3 features) | 28,955 | 28,020 | 0.827 | Underfit |
| RF Underfit (max_depth=2) | 31,883 | 29,762 | 0.735 | Underfit |
| Lasso Underfit (α=10) | 33,404 | 34,843 | 0.718 | Underfit |
| DT Overfit (max_depth=None) | **~0** | 34,557 | 0.689 | **Extreme Overfit** |
| DT Underfit (max_depth=2) | 38,587 | 39,315 | 0.613 | Underfit |
| XGBoost Underfit | 63,531 | 62,032 | 0.112 | **Extreme Underfit** |

### საბოლოო მოდელის შერჩევა

**CatBoost v2** (MLflow Model Registry-ში: `catboost_baseline`, version 3) შეირჩა:
- **Test RMSE** საუკეთესოა Pipeline-ების გათვალისწინებით
- **Train/Test ratio** გონივრულია — არარ არის extreme overfitting
- **Pipeline v2** ბევრად უფრო სწორ preprocessing-ს ახდენს — One-Hot Encoding, Targeted Outlier Removal, Skew Correction

---

## Overfitting vs Underfitting — ძირითადი გაკვეთილები

### Overfitting-ის ნიშნები (High Variance):
- `Train RMSE ≪ Test RMSE`
- მაგ: XGBoost v1: Train=154, Test=22,801
- მაგ: DT Overfit: Train≈0, Test=34,557

### Underfitting-ის ნიშნები (High Bias):
- `Train RMSE ≈ Test RMSE` (ორივე მაღალი)
- მაგ: LR 3 features: Train=28,955, Test=28,020
- მაგ: XGBoost Underfit: Train=63,531, Test=62,032

### რა იწვევს Overfitting-ს?
- მოდელი ძალიან `complex` (Decision Tree max_depth=None)
- Regularization-ის ნაკლებობა (XGBoost reg_alpha=0)
- მცირე training set (Z-Score outlier removal)
- ცუდი encoding (LabelEncoder)

### რა იწვევს Underfitting-ს?
- მოდელი ძალიან `simple` (DT max_depth=2, LR 3 features)
- ძალიან მაღალი Regularization (Lasso α=10, Ridge α=10000)
- სასარგებლო feature-ების ნაკლებობა

---

*ამ README-ი ასახავს ყველა ეტაპს, ყველა ექსპერიმენტს და ყველა გადაწყვეტილებას, რომელიც პროექტის განმავლობაში მიღებულ იქნა.*
