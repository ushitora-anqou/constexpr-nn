# constexpr-nn

## なんこれ

C++17のconstexprを使ってニューラルネットワークを **コンパイル時に** 学習させようというプロジェクト。

## 必要なもの

- C++17が動くコンパイラ
    - まともに動かすためには`clang++`が必要。`g++`だとnewしたメモリをdeleteしてくれない。
    - さらにちゃんと動かすためにはClangに`clang.diff`のパッチを当てたものが必要。Clang内部で使用されている`unsigned int`のサイズ（典型的には32bit）では、実行ステップ数やコールスタックフレームの数が間に合わないため。
- Python3とChainer
- たくさんのメモリ
- そこそこのCPU
- 知恵と勇気と愛と忍耐

## やり方

- `git clone`
- `git submodule init`
- `git submodule update`
- `python mnist.py`
- `python random_table.py`
- `make`
    - 時間がかかる可能性があるので注意。他の仕事をすませると良い。散歩するとか。
- `./main`

