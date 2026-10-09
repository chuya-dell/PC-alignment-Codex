# PC-alignment-Codex Claude Code 用の規約

作業の前に、ユーザーの CLAUDE.md に書かれたノート置き場の2つのファイルを読み、従う。リポジトリの作業規約は `AGENTS.md` にある。

## モデル選択の規則

> 仮置き(2026-10-09、白石の指示)。この節は `PC-alignment-Codex/CLAUDE.md`、`PC-alignment-Codex/AGENTS.md`、Vault の `10_引き継ぎ/00_人工知能の周回規則.md` の3か所で同一にする。変えるときは3か所をそろえる。
> 2026-10-08 の決定(`06_決定/261008_周回のモデル割り当て.md`)のうち「昇格は白石が承認」「進展のない議論が2往復続いたら周回を停止」は、この節の3.と4.に置き換える(2026-10-09 の白石の指示)。優先順位(研究の進展 > 研究の品質 > 継続稼働 > コスト削減)、使わないモデル、研究データの扱いは変えない。
> 使わないモデル:Fable 5.1(クレジットが要る)、GPT-5.6 の世代(gpt-5.6-sol / terra / luna)。

### 作業の種類と最初のモデル(仮置き)
| 作業の種類 | 担当 | 最初のモデル | 呼び方 |
|---|---|---|---|
| 周回を進める本体(指示・検収・通常の作業) | Claude Code | Sonnet 5.5 | `.claude/settings.json` の `model`(`claude-sonnet-5-5`) |
| ログの書き込み、結果の整形、`03_やること`・`02_進捗` の更新、解析結果の題名と親フォルダ名への【未読】の付与 | recorder | Haiku 5.5 | サブエージェント `recorder`(`.claude/agents/recorder.md`)。解析の判断はしない |
| 周回の開始時に検証する仮説と達成条件を決める。終了時に結論が妥当かを判定する。進展のない議論の打開か停止かの判断 | research-lead | Opus | サブエージェント `research-lead`(`.claude/agents/research-lead.md`)。1周回につき3回まで |
| ファイルの列挙、整形、要約 | Codex | GPT-6 Luna / 低 | `codex exec -p light` |
| 通常の解析、コード修正、テスト(既定) | Codex | GPT-6.1 Sol / 中 | `codex exec -p main` |
| main で同じ作業に2回失敗したとき、原因不明の誤差の調査 | Codex | GPT-6.1 Sol / 高 | `codex exec -p deep` |
| research-lead が必要と判断した高度な技術検証 | Codex | GPT-6 Astra / 高 | `codex exec -p verify`。1周回につき1回まで |

Codex の呼び方:必ず `-p <light|main|deep|verify>` を付け、標準入力を閉じる(`< /dev/null`)。閉じないと `codex exec` が入力待ちで止まり続ける(2026-10-09 に実測)。プロファイルは `~/.codex/<名前>.config.toml` で、`~/.codex/config.toml` の既定は main と同じ。

### 規則
1. 作業の種類で、最初のモデルを決める(上の表)。迷う作業は main(Codex)/ Sonnet(Claude Code)から始める。
2. 失敗したら、自動で1段だけ上げてよい(light → main → deep、Haiku → Sonnet)。同じ段で2回失敗したら上げる。「失敗」は、検収で不合格、またはエラー終了。deep と Sonnet より上へは、規則3.の範囲でだけ上げる。
3. Opus(research-lead)と Astra(verify)は、回数の上限(research-lead は1周回につき3回、verify は1周回につき1回)の中なら、白石の承認なしで呼んでよい。上限を超えそうなときは、周回を停止し、停止理由を周回記録に書く。
4. 同じ問題で、進展のない議論が2往復続いたら、research-lead を呼ぶ。research-lead でも打開できなければ周回を停止する。研究上の進展が続いている議論は止めない。
5. 研究データの削除・上書きは、モデルにかかわらず白石の事前承認とする(従来どおり)。
6. 研究の結果に直接効く判断(仮説の採否、結論)を、light や Haiku に任せない。recorder と light は、記録と整形だけを行う。
7. 学習の記録:周回のたびに recorder が、`docs/モデル選択の記録.md`(PC-alignment-Codex)に1行ずつ追記する(日付、作業の種類、最初のモデル、上げたかどうか、上げた理由)。同じ作業の種類で、軽いモデルからの格上げが3回以上記録されたら、その作業の種類の最初のモデルを1段上に書き換えてよい。書き換えたら、この節の表を3か所とも直し、記録に「表を書き換えた」と1行足す。書き換えの対象は light → main と Haiku → Sonnet と main → deep まで。Opus と Astra を最初のモデルにはしない。
