# keirin

## 概要
Pythonを用いて競輪の予測モデルを作成したい。


## 参考サイト
https://qiita.com/kazuktym/items/b52b7a12104bfade03cc#過去の有馬記念のデータで検証
https://qiita.com/GOTOinfinity/items/877fc90168d84d8d1297


## Git操作方法

### リモートからローカルにプルするとき
```sh
$ git pull
```

### ローカルの変更をリモートに反映させたいとき
```sh
$ git add .
$ git commit -m "〇〇〇"
$ git push origin main
```

### ブランチを切って開発するとき


1.mainブランチに移動
```sh
$ git checkout main
```

2.ブランチを派生させる
```sh
$ git checkout -b 〇〇〇
```

### 作業ブランチの内容をmainブランチに反映させる(mergeさせる)方法

*マージする前にローカルの状態を最新にしておく

1.mainブランチに移動する
```sh
$ git checkout main
```

2.mainブランチにマージする
```sh
$ git merge 作業ブランチ名
```