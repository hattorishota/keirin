import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

race_data = pd.read_csv('LR2022race_data.csv', index_col=0)

#concatメソッドで列を分割し，もとの列を削除
race_data = pd.concat([race_data, race_data["選手名府県/年齢/期別"].str.split("/", expand=True)], axis=1).drop("選手名府県/年齢/期別", axis=1)
#選手名の列を削除
race_data = race_data.drop(0, axis=1)
#年齢と期別の列名を変更
race_data = race_data.rename(columns={1: "年齢", 2: "期別"})

#ダミー変数の対象と，カテゴリーを定義
dummy_targets = {"予想": ["nan", "×", "▲", "△", "○", "◎", "注"], \
                      "好気合": ["★"], \
                      "脚質": ["両", "追", "逃"], \
                      "級班": ["A1", "A2", "A3", "L1", "S1", "S2", "SS"] }

#定義したカテゴリーを指定しておく
for key, item in dummy_targets.items():
    race_data[key] = pd.Categorical(race_data[key], categories=item)

#ダミー変数化されたデータフレームを格納するリストと削除する列のリストを定義
dummies = [race_data]
drop_targets = []

#ダミー変数化してdummiesに代入
for key, items in dummy_targets.items():
    dummy = pd.get_dummies(race_data[key])
    dummies.append(dummy)
    drop_targets.append(key)

#ダミー変数化されたデータフレームを大元のデータフレームに結合
race_data = pd.concat(dummies, axis=1).drop(drop_targets,  axis=1)

#落車などで順位が出なかった部分を9位として変換
race_data = race_data.replace(["失", "落", "故", "欠"], 9)

#ギヤ倍数の表示がおかしい部分を変換
race_data["ギヤ倍数"] = race_data["ギヤ倍数"].map(lambda x: x[:4] if len(x)>4 else x)

#期別に含まれる欠車の文字を除外
race_data["期別"] = race_data["期別"].map(lambda x: x.replace(" （欠車）", "") if "欠車"in x else x)

#着順の列を3着以内は1,それ以外は0に変換
race_data["着順"] = race_data["着順"].map(lambda x: 1 if x in ["1", "2", "3"] else 0)

# インデックスを設定（レースだけを特定する場合は、16バイト目までを使用）
race_data['race_index'] = race_data['レースID'].astype(str).str[0:16]
race_data.set_index('race_index', inplace=True)

# 不要なカラムを削除
race_data.drop(['レースID'], axis=1, inplace=True)

#訓練データとテストデータに分ける
X = race_data.drop(["着順"], axis=1)
y = race_data["着順"]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=0)

#アンダーサンプリング
rank_0 = y_train.value_counts()[1]
rank_1 = y_train.value_counts()[1]

rus = RandomUnderSampler(sampling_strategy={0: rank_0, 1: rank_1},random_state=71)

X_train_rus, y_train_rus = rus.fit_resample(X_train.values, y_train.values)

#訓練
model = LogisticRegression()
model.fit(X_train_rus, y_train_rus)

#スコアを表示
print(model.score(X_train, y_train), model.score(X_test, y_test))

#予測結果を確認
y_pred = model.predict(X_test)
pred_df = pd.DataFrame({"pred": y_pred, "actual": y_test})
pred_df[pred_df["pred"] == 1]["actual"].value_counts()
print(pred_df)

# 正解率を表示
print(accuracy_score(y_test, y_pred))

# 混同行列を表示
print(confusion_matrix(y_test, y_pred, labels=[1, 0]))

# 適合率を表示
print(precision_score(y_test, y_pred))

# 再現率を表示
print(recall_score(y_test, y_pred))

# F値を表示
print(f1_score(y_test, y_pred))

#回帰係数の確認
coefs = pd.Series(model.coef_[0], index=X.columns).sort_values()
coefs

#構築したモデルを保存
filename = 'model_sample.pickle'
pickle.dump(model, open(filename, 'wb'))

import pickle

filename = 'model_sample.pickle'
clf = pickle.load(open(filename, 'rb'))

#適当に選んだレースのrace_index
target_race_index = ['3520221228030011']

for idx in target_race_index:
  #適当に選んだレースの説明変数(X_target)と目的変数(y_target)を取得
  X_target = X[X.index == idx] 
  y_target = y[idx]

  #予測
  y_pred_proba = clf.predict_proba(X_target)

  # 辞書に変換(key:馬番, value:1になる確率)
  keys = list(range(3, y_pred_proba[:, 1].size + 1))
  values = y_pred_proba[:, 1]
  pred_dict = dict(zip(keys, values))

  # 結果表示
  print('y=', idx[0:4])
  print(dict(sorted(pred_dict.items(), key=lambda x:x[1], reverse=True)))