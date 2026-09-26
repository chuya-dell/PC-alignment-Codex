# AI開発品質保証プロトコル

対象リポジトリ: PC-alignment-Codex
このファイルはClaude Codeへの動作指示書として機能する。人間向けの仕様書であると同時に、Claude Codeはこの内容に従って自律的にオーケストレーションを行う。

## 目的

Codexを実装者、Claudeを独立レビュアーとして分離し、両者の間の受け渡し作業(差分の取得・レビュー依頼・指摘の伝達・修正の反映・テストの実行・結果の突き合わせ)を人間が手動で仲介する必要をなくす。人間が担う作業は、最初の要件提示と、Human Gate到達時の最終判断のみに限定する。

## 最上位原則

誰が何を根拠に判断したかを、後から再構成できること。

この仕組みの目的は「AIを2つ使えば精度が上がる」ことではなく、AIの個々の判断を信用しなくても運用できる開発プロセスを作ることである。

## 基本原則

1. 実装とレビューを分離する。同一のモデルに実装させて自分自身にレビューさせない。
2. レビューの指摘を盲目的に採用しない。Codexは指摘を検証し、妥当なものだけを修正する。却下する場合は理由を必ず記録する。
3. 自然言語のやり取りそのものを中心にしない。共通の事実として扱うのはコード、差分、テスト結果、要件である。
4. テストを最終的な事実として重視する。ただしテスト自体が要件を網羅しているかどうかも監査対象に含める。
5. レビュー往復の回数に上限を設ける。
6. 最終的な意思決定は人間に残す。

## 役割定義

- **Codex**: 実装者。要件に基づきコードとテストを書き、レビュー指摘を検証して受諾・却下を判断する。
- **Claude(Claude Code)**: 独立レビュアー、かつオーケストレーター。差分を独立した立場でレビューし、同時にCodexの呼び出し・記録・状態判定などの受け渡し作業を担う。
- **突き合わせ処理**: 決定的な機械的処理(スクリプト)。前回の指摘と今回の状況を機械的に突き合わせ、解消・再発・新規・見逃しに分類する。この処理はどちらのモデルにも委任しない。判断の余地がある処理ではなく、ファイル名・行範囲・要件番号の一致を確認するだけの決定的な処理にする。
- **人間(忠弥)**: 要件を提示し、Human Gateに到達した論点について最終判断を下す。

Reviewer(独立レビュアー)と突き合わせ処理の担当者は別にする。再レビューを「前回の指摘が直ったか確認して」という形にすると、レビューが修正確認作業に矮小化されるため、毎回コード全体(または該当モジュール全体)を独立して評価し直し、前回の指摘との突き合わせは別の機械的な処理に任せる。

## リポジトリ構成

```
project/
├── requirements/
│   ├── requirements.md       要件本体。変更のたびにコミットする
│   └── traceability.md       要件番号とテストケースの対応表
│
├── src/
├── tests/
│
└── .ai-review/
    ├── config.yaml
    ├── review-log.md         人間向けの要約ログ
    ├── escalation.md         Human Gate到達時にのみ生成される
    └── rounds/
        ├── 001/
        │   ├── claude-review.json
        │   ├── codex-response.json
        │   ├── test-results.json
        │   └── reconciliation.json
        └── 002/
```

## 状態遷移

| 状態 | 入る条件 | 出る条件 | 生成物 |
|---|---|---|---|
| REQUIREMENTS_DRAFT | 新規要件の提起 | requirements.mdがコミットされる | requirements.md、要件バージョン番号 |
| IMPLEMENTING | 要件バージョンが確定 | Codexが実装・テストを完了しコミット | src/、tests/、コミットハッシュ |
| AWAITING_REVIEW | 実装コミットが存在 | Claudeにレビュー依頼が渡る | 差分 |
| REVIEWING | レビュー依頼を受理 | claude-review.jsonを出力 | rounds/00N/claude-review.json |
| RESPONDING | 指摘一覧を受理 | 全指摘に対し受諾・却下の判断が記録される | rounds/00N/codex-response.json |
| REVISING | 受諾された指摘が存在 | 修正コミットが作成される | 新しいコミットハッシュ |
| TESTING | 修正コミットが存在 | テスト実行完了 | rounds/00N/test-results.json |
| RECONCILING | 前Roundの指摘と現状が揃う | 突き合わせ処理による分類完了 | rounds/00N/reconciliation.json |
| ROUND_COMPLETE | 上記すべて完了 | 次の分岐判定へ | review-log.mdへの追記 |
| APPROVED | 指摘全解消・テスト全通過・対応表完備 | (終端) | - |
| ESCALATED | 下記のいずれかに該当 | Human Gateへ | escalation.md |

## Round完了条件

- 全指摘に受諾・却下の判断が記録されている
- 該当コミットのテストが実行済み
- 突き合わせ処理による分類が完了している

## 次Roundへ進む条件

- 未解決(再発または新規)の指摘が残っている
- Round数が上限(3)未満

## APPROVED条件

- 指摘がゼロ、または全て解消済み
- テスト全通過
- 要件とテストの対応表に漏れがない

## ESCALATED条件(Human Gate)

- Round数が上限(3)に到達した
- 同一の指摘(同一の該当箇所と要件番号の組み合わせ)が2Round連続で却下され続けた
- Codexの却下理由が3回以上同一パターンで繰り返されている(却下理由の使い回しの兆候)
- 重大度が高く、かつ安全性に関わる指摘が発生した場合はRound数の消費を待たず即座にエスカレーションする
- 指摘の内容が要件そのものの矛盾・曖昧性を指している場合(CodexにもClaudeにも判断権限がない領域のため)

## ファイルスキーマ

### claude-review.json

```json
{
  "round": 1,
  "reviewed_commit": "abc123",
  "requirements_version": "v3",
  "findings": [
    {
      "id": "F-001",
      "severity": "high",
      "category": "correctness",
      "location": { "file": "src/xxx.py", "lines": [45, 50] },
      "requirement_ref": "R-01",
      "description": "...",
      "evidence": "..."
    }
  ],
  "traceability_gaps": [
    { "requirement_id": "R-02", "covered_by_test": false }
  ]
}
```

### codex-response.json

```json
{
  "round": 1,
  "responses": [
    {
      "finding_id": "F-001",
      "decision": "reject",
      "rationale": "呼び出し元で検証済みのため冗長",
      "commit_if_accepted": null
    }
  ]
}
```

### test-results.json

```json
{
  "round": 1,
  "commit": "def456",
  "passed": 40,
  "failed": 2,
  "failed_tests": ["test_xxx"],
  "requirement_coverage": {
    "R-01": ["test_empty_input"],
    "R-02": []
  }
}
```

### reconciliation.json

突き合わせ処理(決定的なスクリプト)の出力。指摘の識別番号と該当箇所の機械的な突き合わせのみを行う。

```json
{
  "round": 2,
  "classification": [
    { "finding_id": "F-001", "status": "resolved" },
    { "finding_id": "F-002", "status": "recurred" },
    { "finding_id": "F-003", "status": "new" }
  ]
}
```

## config.yaml

```yaml
max_rounds: 3
escalation_triggers:
  same_rejection_pattern_count: 3
  high_severity_security_finding: true
  requirement_ambiguity_flagged: true
requirement_traceability_required: true
reconciler: deterministic_script   # モデルに委任しない
```

## Codexの呼び出し方法

Claude Codeはbashツールから`codex exec`を非対話モードで呼び出す。対話画面は開かない。

- 実装・修正の実行時は`codex exec --sandbox workspace-write "<プロンプト>"`を使う。リポジトリ内での書き込みのみを許可し、ネットワークアクセスを伴う操作は許可しない
- 進捗の把握には`--json`を付け、標準出力に流れるJSON Linesを解析する。ターン完了・ターン失敗のイベントを見て、実装が終わったかどうかを機械的に判定する。人間が目視で完了を確認する必要はない
- 認証鍵は環境変数として常時設定せず、単発の呼び出しごとにのみ渡す。リポジトリ内のコード(Codexが生成したものを含む)が鍵を読み取れる状態を作らない

## Claude Code(オーケストレーター)の動作フロー

1. 要件(requirements.md)を確認し、要件バージョンを控える
2. `codex exec --sandbox workspace-write --json "<要件に基づく実装指示>"`を実行する
3. 完了後、差分を取得する
4. その差分を独立レビュアーとして評価し、claude-review.jsonを出力する
5. findingsをCodexに渡し、`codex exec`で受諾・却下の判断と理由を書かせ、codex-response.jsonに記録する
6. 受諾された指摘についてCodexに修正を実行させ、新しいコミットを作る
7. テストを実行し、test-results.jsonに記録する
8. 突き合わせ処理(スクリプト)を実行し、reconciliation.jsonを出力する
9. Round完了条件を満たしたらreview-log.mdに要約を追記する
10. APPROVED条件を満たせば終了。ESCALATED条件に該当すればescalation.mdを生成し、そこで人間の判断を待つ。どちらでもなければ2に戻る

## escalation.mdのひな形

```
## 未解決の論点まとめ

### 論点1
- Claudeの主張: ...
- Codexの反論: ...
- 現状のコード該当箇所: xxx.py 120-135行目

### 現在のテスト結果
- 通過: ...
- 未通過: ...

### 要求される判断
上記のうちどちらの立場を採用するか、もしくは別案が必要か。
```

## 運用上の注意

- 再レビューの際、前回の指摘文はCodexにもClaude自身にも見せず、対象コード全体(または該当モジュール全体)をゼロから評価する。前回の指摘との突き合わせは手順8の機械的な処理にのみ任せる
- 却下理由は必ずrationaleとして記録し、次回のレビュー対象に含める。却下理由が形骸化していないかどうかも監査対象とする
- レビュー対象は実装コードだけでなく、テストコードが要件をどこまで網羅しているかも含める
- 要件そのものが変更された場合は、requirements.mdをコミットし、その時点のコミットハッシュまたはバージョン番号をCodexへの指示とClaudeのレビュー依頼の両方に明記する
