# Technical Writing Cheatsheet

Code is read by machines; documentation is read by humans. Both require clarity, structure, and optimization.
Based on *Google Developer Documentation Style Guide*, *The Elements of Style*, and *理科系の作文技術 (Technical Writing for Science and Engineering)*.

## 1. Core Principles (The "Linting" Rules)

### Active Voice
Make it clear *who* is performing the action. Passive voice obscures responsibility and bloats text.

*   **Bad**: The data is processed by the function.
*   **Good**: The function processes the data.
*   **Bad**: It is recommended to use JAX for acceleration.
*   **Good**: We recommend using JAX for acceleration. / Use JAX for acceleration.

### Second Person (Use "You")
Address the user directly. Third-person terms like `User` or `Developer` create unnecessary distance.

*   **Bad**: The user needs to install dependencies first.
*   **Good**: Install dependencies first. / You must install dependencies.

### Present Tense
Documentation describes what exists *now*. Avoid future tense (`will`) unless promising a future feature.

*   **Bad**: The script will run the simulation.
*   **Good**: The script runs the simulation.
*   **Bad**: Running this command will cause an error.
*   **Good**: Running this command causes an error.

---

## 2. Structure & Layout (Refactoring Text)

### BLUF (Bottom Line Up Front)
State the conclusion first. Readers are busy; do not bury the lead.

*   **Paragraph**: The first sentence (Topic Sentence) should summarize the paragraph.
*   **Document**: The first section should explain "What is this?" and "Why does it matter?".

### Lists over Prose
Break down complex series (3+ items) into lists.

**Bad**:
To optimize the model, you can adjust the learning rate, change the batch size, and increase the number of epochs.

**Good**:
To optimize the model:
*   Adjust the learning rate.
*   Change the batch size.
*   Increase the number of epochs.

### Parallelism in Lists
Ensure all list items start with the same part of speech (usually imperative verbs).

*   **Bad**:
    *   Download the file.
    *   Configuration of the server.
    *   To start the app, run `main.py`.
*   **Good**:
    *   Download the file.
    *   Configure the server.
    *   Start the app by running `main.py`.

---

## 3. Clarity & Conciseness (Optimization)

### Kill Filler Words
Delete words that add no meaning. Treat them like compiler warnings.

| Remove | Why |
| :--- | :--- |
| **basically** / **essentially** | Adds no information. |
| **very** / **really** / **quite** | Weakens adjectives. Use stronger words or specific numbers. |
| **in order to** | "to" is sufficient. |
| **at this point in time** | "now" or "currently" is sufficient. |
| **needless to say** | If it's needless, don't say it. |

### Short Sentences
One idea per sentence. Long sentences breed ambiguity. Break them up.

*   **Bad**: The API is fast, but it is experimental, so you should use it with caution because it might change.
*   **Good**: The API is fast but experimental. Use it with caution; it might change.

### Clarify Antecedents
Avoid vague `it`, `this`, `that`.

*   **Bad**: Python is slow. JAX compiles it to XLA. **This** makes it faster.
*   **Good**: Python is slow. JAX compiles Python to XLA. **This compilation** makes execution faster.

---

## 4. Specific Artifacts Templates

### README.md Structure
The landing page. Answer "What is this?" in 3 seconds.

1.  **Project Name & One-line Description**: What is this?
2.  **Badges**: CI status, License, Version.
3.  **Why this?**: Differentiators, value proposition.
4.  **Quick Start**: Minimal installation and "Hello World" code.
5.  **Installation**: Detailed setup instructions.
6.  **Usage/Examples**: Common use cases.

### Git Commit Messages (Conventional Commits)
Machine-parseable history.

Format: `<type>(<scope>): <subject>`

*   `feat`: New feature
*   `fix`: Bug fix
*   `docs`: Documentation only
*   `style`: Formatting (whitespace, semi-colons, etc.)
*   `refactor`: Code change that neither fixes a bug nor adds a feature
*   `perf`: A code change that improves performance
*   `test`: Adding missing tests or correcting existing tests
*   `chore`: Build process or auxiliary tool changes

**Example**:
`feat(solver): implement sparse matrix support for GMRES`

---

## 5. Tools (Linter for Prose)

Just as code has linters, prose has linters.

*   **[Vale](https://vale.sh/)**: CLI-based prose linter. Enforces style guides (Google, Microsoft) in CI.
*   **[Hemingway Editor](https://hemingwayapp.com/)**: Visualizes sentence complexity. Highlights passive voice and adverbs.
*   **[Grammarly](https://www.grammarly.com/)**: Standard grammar and spell checker.

---

## 6. Common Typos & Gotchas in Tech

*   **Setup** (noun) vs **Set up** (verb)
    *   "Check the **setup**." / "Please **set up** the env."
*   **Login** (noun/adj) vs **Log in** (verb)
    *   "Login page" / "Please log in."
*   **i.e.** (that is) vs **e.g.** (for example)
    *   Often confused. If unsure, use English words ("that is", "for example").
*   **data**
    *   Technically plural, but accepted as singular (uncountable) in modern usage. Be consistent.
    *   "The data **is** processed." (Preferred)

---

## 7. 理科系の作文技術

木下是雄著『理科系の作文技術』に基づく。

### 7.1. 基本姿勢

#### 事実と意見の区別
最も重要な原則。「事実（Fact）」と「意見（Opinion）」を厳密に区別する。

*   **事実**: 証拠を示してその真偽を客観的に確認できる記述。自然現象、実験結果、数式など。
    *   *例*: 「AとBを混合すると爆発した。」
*   **意見**: 推論、判断、感想、仮説。
    *   *例*: 「AとBの混合は**危険である**。」（"危険"は判断）
*   **ルール**: 事実の記述の中に、主観的な形容詞（美しい、悲惨な、画期的な）を混ぜない。「〜であると思う」と「〜である」を使い分ける。

#### 読者の設定
「誰に読ませるか」を最初に決める。
*   特定の専門家向けか？ 一般向けか？
*   読者が持っている前提知識は何か？
*   **ルール**: 読者が知りたい情報を、読者が理解できる言葉で書く。自己満足の日記ではない。

### 7.2. 構成

#### 起承転結の否定
理系の文章に「起承転結」は不要であり、有害ですらある。
*   **文学**: クライマックス（転・結）まで結論を隠す。
*   **理系**: **結論が先**（Conclusion First）。

#### 重点先行主義
1.  **目標規定文**: この文書は何を主張するものか、最初に一文で定義する。
2.  **要約（Summary）**: 結論と重要ポイントを最初に書く。
3.  **本論**: 詳細な論証。
4.  **結論**: まとめ。

### 7.3. パラグラフ

*   **一トピック一パラグラフ**: 1つのパラグラフには1つのトピックだけを入れる。
*   **トピックセンテンス**: パラグラフの最初の文（トピックセンテンス）で、その段落の要約や主張を述べる。残りの文はその展開や裏付け。
*   **接続詞**: 接続詞（しかし、したがって）を使って、パラグラフ間の論理的なつながりを明確にする。

### 7.4. 文体

#### 明確な主語と述語
日本語は主語を省略しがちだが、技術文では誤解を招く。
*   **Bad**: データを入力するとエラーになった。（何が？誰が？）
*   **Good**: ユーザーがデータを入力すると、システムはエラーを返した。

#### 簡潔な表現
*   **逆茂木（さかもぎ）文を避ける**: 修飾語が長すぎて主語になかなかたどり着かない文。
    *   *Bad*: 非常に高速で、メモリ効率も良く、最新のGPU向けに最適化された**ライブラリ**。
    *   *Good*: この**ライブラリ**は高速かつ省メモリで、最新のGPUに最適化されている。
*   **無意味な繋ぎ言葉を削る**: 「〜における」「〜に関する」「〜を行う」などは多くの場合削除できる。
    *   *Bad*: データの解析**を行う**。
    *   *Good*: データを解析する。
    *   *Bad*: パフォーマンス**に関する**問題。
    *   *Good*: パフォーマンスの問題。

#### 漢字と平仮名のバランス
*   難しい漢字を使いすぎない。
*   接続詞や副詞は平仮名にする（例：「〜の**ため**」「**かつ**」「**また**」）。
*   「書けるけれど書かない」勇気を持つ。
