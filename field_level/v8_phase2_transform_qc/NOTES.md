# Phase 2 affine transform QC gate

## 実装内容

`shared.registration.assess_affine_transform_qc` を、保存済み382組比較CSVの回帰検証に使う。
CSVには元行列ではなく中心変位・回転・等方スケールだけが保存されているため、本スクリプトは
同一の中心変位を持つ相似アフィンを再構成してQC関数を評価する。生画像の再登録は行わない。

## 結果概要と保存先

実行結果はGit管理外の指定`--out-dir`に、行単位CSVと要約CSVとして保存する。2026-09-18の
結果はGoogle DriveおよびObsidian `06_解析`に保存する。

## 既知の問題

CSVに記録された旧150 px/10%退化条件は6組であり、関連レポートにある8組とは母集団が異なる。
元アフィン行列は保存されていないため、shear/anisotropyの履歴回帰はできない。
