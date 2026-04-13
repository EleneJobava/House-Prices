# House Prices - Advanced Regression Techniques

## Kaggle-ის კონკურსის მოკლე მიმოხილვა

ეს პროექტი შესრულებულია Kaggle-ის კონკურსისთვის **House Prices: Advanced Regression Techniques**. ამ ამოცანის მიზანია აიოვას შტატში არსებული სახლების ფასის პროგნოზირება 79-მდე მახასიათებლის გამოყენებით.  
მთავარი აქცენტი გავაკეთე არა მხოლოდ მაქსიმალურ ქულაზე, არამედ პროცესზე: როგორ იცვლება შედეგი, როცა იცვლება `Cleaning`, `Feature Engineering`, `Feature Selection` და მოდელის ჰიპერპარამეტრები.

---

## ჩემი მიდგომა პრობლემის გადასაჭრელად

სამუშაო ეტაპობრივად ავაწყე, რომ თითოეული ექსპერიმენტი გასაგები და ერთმანეთთან შედარებადი ყოფილიყო:

1. საწყისი EDA და baseline მოდელი (`Linear Regression`) — რათა მქონოდა საწყისი საზომი წერტილი.
2. მონაცემთა გაწმენდა (NaN და outlier-ების დამუშავება) და გავლენის გაზომვა RMSE/R2-ზე.
3. ახალი feature-ების შექმნა და categorical ცვლადების სწორი კოდირება.
4. სხვადასხვა ტიპის მოდელების ტესტირება: ხაზოვანი, ხეებზე დაფუძნებული და boosting.
5. overfit/underfit შემთხვევების მიზანმიმართული დემონსტრირება და ანალიზი.
6. ყველა მნიშვნელოვანი run-ის დალოგვა MLflow-ზე (DagsHub tracking URI-ით).
7. საუკეთესო მოდელის Model Registry-ში შენახვა და inference notebook-ში მისი გამოყენება.

---

## რეპოზიტორიის სტრუქტურა

`House-Prices/`

- `README.md` — პროექტის სრული აღწერა.
- `model_experiment.ipynb` — EDA, preprocessing, feature engineering/selection, model training და MLflow logging.
- `model_inference.ipynb` — Model Registry-დან საუკეთესო მოდელის ჩატვირთვა, test set-ზე პროგნოზი, `submission.csv` გენერაცია.
- `feature_names.json` — training-ის დროს გამოყენებული feature-ების სია ინფერენსისთვის.
- `train_columns.json` — train/test alignment-ისთვის საჭირო სვეტების სია.
- `skewed_features.json` — skewed ცვლადების სია, რომლებსაც `log1p` ტრანსფორმაცია უტარდებათ.
- `lot_frontage_medians.json`, `garage_area_medians.json` — neighborhood-ზე დაფუძნებული median მნიშვნელობები inference pipeline-სთვის.
- `submission.csv` — Kaggle-ზე ასატვირთი პროგნოზები.

---

## Feature Engineering

### კატეგორიული ცვლადების რიცხვითში გადაყვანა

თავდაპირველად გამოყენებული `LabelEncoder` ნომინალურ კატეგორიებზე (მაგ. `Neighborhood`) ქმნიდა ხელოვნურ რიგითობას, რაც განსაკუთრებით აზიანებდა ხეებზე დაფუძნებულ მოდელებს.  
შემდეგ ეტაპზე მიდგომა გავყავი ორ ნაწილად:

- ხარისხობრივ (ordinal) სვეტებზე — ხელით განსაზღვრული mapping (`Po < Fa < TA < Gd < Ex`);
- ნომინალურ (nominal) სვეტებზე — `One-Hot Encoding`.

ამ ცვლილებამ მოდელებს მისცა უფრო სწორი სიგნალი და განსაკუთრებით გააუმჯობესა არაწრფივი მოდელების სტაბილურობა.

### NaN მნიშვნელობების დამუშავება

NaN-ების შევსება ერთი წესით არ გამიკეთებია, რადგან სვეტების სემანტიკა განსხვავდება:

- `"None"` fill იქ, სადაც NaN ნიშნავს მახასიათებლის რეალურ არარსებობას (მაგ. აუზი, ღობე, ბუხარი);
- mode fill კატეგორიულ სვეტებზე მცირე missing rate-ის შემთხვევაში;
- `LotFrontage` და `GarageArea` — `Neighborhood` მიხედვით median fill (ლოკაციის მიხედვით უფრო რეალისტური);
- ზოგიერთ რიცხვით სვეტზე zero fill, როცა ცარიელი მნიშვნელობა ფაქტობრივად “არ აქვს” შემთხვევას ასახავს.

ეს იყო ერთ-ერთი ყველაზე ეფექტური ნაბიჯი feature engineering-ის ბლოკში.

### დამატებული feature-ები

ფასზე პირდაპირი გავლენის გასაძლიერებლად შევქმენი აგრეგირებული და ინტერაქციის ტიპის ცვლადები, მაგალითად:

- `TotalArea`, `TotalBaths`, `TotalPorch`;
- `HouseAge`, `RemodAge`;
- `OverallScore`, `LivArea_Qual`, `Qual_TotalArea`;
- არსებობის ბინარული ინდიკატორები (`Pool`, `Garage`, `Bsmt`, `Fireplace`);
- `MoSold`-ის ციკლური ენკოდინგი (`sin/cos`), რათა დეკემბერი და იანვარი “შორს” არ აღმოჩნდეს.

ამ ეტაპმა baseline ხაზოვან მოდელზე RMSE მნიშვნელოვნად შეამცირა და R2 გაზარდა.

---

## Cleaning მიდგომები

Outlier-ებზე ორი მიდგომა დავტესტე:

1. **Z-score threshold=3** — აგრესიული ფილტრაცია, რომელმაც ხაზოვან მოდელებზე სწრაფი გაუმჯობესება მისცა, მაგრამ ზედმეტად ბევრი ჩანაწერი ამოიღო.
2. **Targeted outlier removal** — domain-ცოდნაზე დაფუძნებული შერჩევა (`GrLivArea`, `LotArea`, `SalePrice` ანომალიები), რომელმაც მონაცემების დიდი ნაწილი შეინარჩუნა და უკეთ იმუშავა არაწრფივ მოდელებზე.

დასკვნა: ზოგადი Z-score კარგია საწყისი გაწმენდისთვის, მაგრამ ამ dataset-ზე საბოლოოდ უფრო ჯანსაღი იყო targeted მიდგომა.

---

## Feature Selection

Feature selection-ზე გამოვცადე რამდენიმე სტრატეგია:

- `SelectKBest (f_regression)` — სწრაფი და მარტივი, მაგრამ ხშირად აგდებდა იმ ცვლადებს, რომლებიც ინდივიდუალურად სუსტი, თუმცა კომბინაციაში მნიშვნელოვანი იყო.
- მოდელზე დაფუძნებული შერჩევა (`SelectFromModel`, tree-based importance) — არაწრფივი კავშირების უკეთ აღქმა.
- რეგულარიზებული მოდელები (`Ridge`, `Lasso`) — ბევრ შემთხვევაში feature selection-ის ფუნქცია თავად მოდელმა შეასრულა.

შედეგად მივიღე, რომ fixed `k`-ზე აგებული selection ყოველთვის არ არის კარგი ამოცანისთვის; pipeline-სა და მოდელზე მორგებული შერჩევა უფრო შედეგიანია.

---

## Training

### ტესტირებული მოდელები

ტრენინგის ეტაპზე გატესტილია:

- `Linear Regression`
- `Ridge`, `Lasso` (manual + `GridSearchCV`)
- `Decision Tree`
- `Random Forest`
- `XGBoost`
- `CatBoost`

### Hyperparameter ოპტიმიზაციის მიდგომა

ჰიპერპარამეტრებზე გამოვიყენე როგორც ხელით შერჩეული boundary შემთხვევები (განზრახ underfit/overfit დემონსტრაციისთვის), ისე `GridSearchCV`:

- ხაზოვან მოდელებზე მთავარი აქცენტი იყო `alpha`-ს გავლენა;
- tree/ensemble მოდელებზე — `max_depth`, `min_samples_*`, `n_estimators`;
- boosting მოდელებზე — `learning_rate`, `depth`, `iterations/estimators`, რეგულარიზაცია.

### Underfit და overfit ანალიზი

პროექტში სპეციალურად დავაფიქსირე ორივე სცენარი:

- **Underfit მაგალითები:** ძალიან მაღალი რეგულარიზაცია (`Ridge alpha=10000`, `Lasso alpha=10`), ზედმეტად შეზღუდული ხეები (`max_depth=2`).
- **Overfit მაგალითები:** დაურეგულირებელი ღრმა ხეები (`DecisionTree max_depth=None`), ზედმეტად ძლიერი boosting კონფიგურაციები.

ანალიზის ძირითადი ინდიკატორი იყო `Train RMSE` და `Test RMSE` შორის განსხვავება:
- train და test ორივე მაღალი და ახლოა -> მაღალი bias (underfit);
- train ძალიან დაბალი, test მკვეთრად მაღალი -> მაღალი variance (overfit).

### საბოლოო მოდელის შერჩევის დასაბუთება

`model_inference.ipynb`-ში საუკეთესო მოდელი Model Registry-დან იტვირთება როგორც `catboost_baseline` (version 3).  
საბოლოო არჩევანი გაკეთდა იმიტომ, რომ ამ მოდელმა preprocessing pipeline-თან ერთად აჩვენა საუკეთესო კომბინაცია:

- ძლიერი პროგნოზის სიზუსტე,
- უკეთესი განზოგადება,
- პრაქტიკულად გამოყენებადი inference flow (`registry -> predict -> submission`).

---

## MLflow Tracking

### MLflow ექსპერიმენტების ბმული

Tracking server (DagsHub):  
**[https://dagshub.com/ejoba22/House-Prices.mlflow](https://dagshub.com/ejoba22/House-Prices.mlflow)**

### ჩაწერილი მეტრიკების აღწერა

ექსპერიმენტების დროს ილოგება:

- მოდელის ტიპი და pipeline ვერსია,
- preprocessing და encoding სტრატეგიები,
- ჰიპერპარამეტრები,
- `train_rmse`, `test_rmse`, `r2`,
- საჭირო არტეფაქტები (feature სია, preprocessing კონფიგები),
- საბოლოო მოდელის რეგისტრაცია Model Registry-ში.

ეს საშუალებას მაძლევს ნებისმიერი შედეგი რეპროდუცირებადი იყოს და მკაფიოდ ვნახო, რომელი ცვლილება რეალურად აუმჯობესებს მოდელს.

### საუკეთესო run-ის შედეგები

საუკეთესო run არჩეულია MLflow comparison-ით და გამოყენებულია inference ეტაპზე.  
`model_inference.ipynb` პირდაპირ იტვირთავს რეგისტრირებულ მოდელს და ქმნის Kaggle submission ფაილს იგივე preprocessing ლოგიკით, რაც training-ში იყო გამოყენებული.

---

## დავალების შეფასების კრიტერიუმებთან შესაბამისობა

- **Feature Engineering (10%)** — სხვადასხვა ტიპის feature-ების შექმნა და მათი გავლენის ანალიზი.
- **Feature Selection (10%)** — რამდენიმე მეთოდის შედარება და უარყოფითი/დადებითი ეფექტების ახსნა.
- **Training (30%)** — მრავალმოდელური ექსპერიმენტები, hyperparameter tuning, overfit/underfit შემთხვევების დემონსტრირება.
- **MLflow Tracking (30%)** — run-ების სისტემური ლოგირება DagsHub-ზე და model registry გამოყენება.
- **Repository Structure (20%)** — მოთხოვნილი ფაილები, ცალკე experiment და inference workflow.

---

## შეჯამება

ამ პროექტში მთავარი ფოკუსი იყო სწორი ექსპერიმენტული პროცესი: ერთი და იგივე მონაცემებზე სხვადასხვა preprocessing და modeling არჩევანის გავლენის შეფასება, შეცდომების ანალიზი და ყველა შედეგის traceability MLflow-ში.  
ფინალურ pipeline-ში მივიღე სტაბილური, გამეორებადი workflow — training notebook-დან დაწყებული registry-ზე შენახვით და inference notebook-ით დასრულებული.
