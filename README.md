# constexpr-nn

## なんこれ

C++17のconstexprを使ってニューラルネットワークを **コンパイル時に** 学習させようというプロジェクト。

## 必要なもの

- C++17が動くコンパイラ
    - まともに動かすためには`clang++`が必要。`g++`だとnewしたメモリをdeleteしてくれない。
        - どうやら`g++`は`constexpr`指定された関数をメモ化するようだ。
    - さらにちゃんと動かすためにはClangに`clang.diff`のパッチを当てたものが必要。Clang内部で使用されている`unsigned int`のサイズ（典型的には32bit）では、実行ステップ数やコールスタックフレームの数が間に合わないため。詳しくは後述。
- Python3とChainer
- たくさんのメモリ
- そこそこのCPU
- 知恵と勇気と愛と忍耐

## Clangにパッチを当てる

基本的には[公式ドキュメント](http://llvm.org/docs/GettingStarted.html#getting-started-quickly-a-summary)を参考にすれば良い。途中で`svn patch clang.diff`を行う。以下典型的な手順。

```
cd where-you-want-llvm-to-live
svn co http://llvm.org/svn/llvm-project/llvm/trunk llvm
cd llvm/tools
svn co http://llvm.org/svn/llvm-project/cfe/trunk clang
svn patch where-you-cloned-constexpr-nn/clang.diff
cd ../..
mkdir build
cd build
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release ../llvm
make -j6
```

パッチを当てたClangを使うように`Makefile`を書き換える必要がある。

## 設定

いくつかユーザーによって設定を変えるべき場所がある。

- エポック数・訓練サンプルのバッチサイズ・テストサンプルのバッチサイズ。
    - `main.cpp`の`main`関数冒頭。
    - `constexpr size_t epoch = 1, train_batch_size = 100, test_batch_size = 100;`
- 訓練・テストに使用するサンプルの数。
    - `mnist.py`
    - `train_sample_num = 15000`
    - `test_sample_num = 15000`
    - 最大60000
- テストをコンパイル時に行うか、実行時に行うか。
    - `main.cpp`
    - `//#define CNSTNN_GET_TEST_ACC_AT_COMPILE_TIME`をコメントアウトすると、コンパイル時に行う。

## あそびかた

- `git clone`
- `git submodule init`
- `git submodule update`
- `python mnist.py`
- `python random_table.py`
- `make`
    - 時間がかかる可能性があるので注意。他の仕事をすませると良い。散歩するとか。
    - Intel Core i7-3630QMで、8000サンプル（train 4000 + test 4000）を1 epochでおよそ13時間。
- `./main`
    - 訓練時のlossとテスト時のaccuracyが出力される。
        - ただしテスト時のaccuracyを出力させるためには`//#define CNSTNN_GET_TEST_ACC_AT_COMPILE_TIME`のコメントアウトが必要。
    - `./main filename`とすると外部から画像を読み込んで、その画像を判定する。


