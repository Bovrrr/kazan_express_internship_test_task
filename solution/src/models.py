from sklearn.base import ClassifierMixin
from catboost import CatBoostClassifier, Pool


class cbClassifier(ClassifierMixin):
    # я решил оборачивать все свои модели(катбус/лгбм/полносвязные сеточки) в шаблоны моделей и трансформаций данных склёрна
    # чтобы можно было легко собирать всё в пайплайны и ансамблировать модели(стекинг/блендинг)

    def __init__(self, params):
        # super.__init__()
        import torch

        self.params = params

        if torch.cuda.is_available():
            self.params["task_type"] = "GPU"
            self.params["devices"] = "0"

        self.model = CatBoostClassifier(**self.params)

    def fit(self, X, y, cat_feats=[], text_feats=[]):
        from sklearn.model_selection import train_test_split
        from catboost import Pool
        from joblib import Parallel, delayed

        # делю на трейн и валидацию, чтоб можно было выбрать непереобученную на трейн модель
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
        # выкидываю пары объект-таргет, где таргет не встречался в терйне
        u_y_train = set(y_train)

        idxs = Parallel(n_jobs=-1)(
            delayed(lambda elem: elem in u_y_train)(y_val[i]) for i in range(y_val.size)
        )
        X_val = X_val[idxs]
        y_val = y_val[idxs]

        # оборачиваю данные в пулы для катбуста
        train_pool = Pool(
            X_train,
            y_train,
            cat_features=cat_feats,
            text_features=text_feats,
        )
        val_pool = Pool(
            X_val,
            y_val,
            cat_features=cat_feats,
            text_features=text_feats,
        )
        # обучение модели
        self.model.fit(
            train_pool,
            eval_set=val_pool,
            use_best_model=True,
            early_stopping_rounds=150,
        )

        return self

    def predict(self, X):
        return self.model.predict(X, task_type=self.params.get("task_type", "CPU"))

    def predict_proba(self, X):
        return self.model.predict_proba(
            X, task_type=self.params.get("task_type", "CPU")
        )
