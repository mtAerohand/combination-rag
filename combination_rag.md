# Combination RAG

- [リランキングモデルによる RAG の日本語検索精度の向上](https://developer.nvidia.com/ja-jp/blog/rag-with-sota-reranking-model-in-japanese/)

## 雑感

- 普通の類似度検索で大まかにドキュメントを取得した後にリランキングモデルで並べ替え
- 精度のみを考えるならリランキングモデルでドキュメントを絞り込むべきだが運用上の問題がある
  - 通常の類似度検索はあらかじめドキュメントを埋め込みしておけるのがデカい
- ので、レイテンシと精度の中間を取ってハイブリッドの retriever にする
- 結局手で実装するならチャンキングがムズい

## 疑問

- Knowledge Base 利用時はリランキングモデル指定しないが、ハイブリッドでやってるのか？RAG ワークフローの詳細が知りたい。ワンチャン自分で実装した方が精度出たりするのでは。
- Kendora は retrieve 部分がブラックボックスだから分からん
