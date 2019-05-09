# Play with MNIST
# MNIST 가지고 놀아보자!

A detailed MNIST walk-through!
자세한 MNIST 워크 스루!

Let's start by loading MNIST from **keras.datasets** and preprocessing to get rows of normalized 784-dimensional vectors.
**keras.datasets** 로부터 MNIST 를 불러오고 전처리를 통해 784 차원의 노멀라이즈된 행 벡터를 만듭시다.


```python
import  tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics

(xs, ys),_ = datasets.mnist.load_data()
print('datasets:', xs.shape, ys.shape, xs.min(), xs.max())

xs = tf.convert_to_tensor(xs, dtype=tf.float32) / 255.
db = tf.data.Dataset.from_tensor_slices((xs,ys))
db = db.batch(32).repeat(10)
```

<img src="mnist.gif" align="right" width="270" height="270">

Now let's build our network as a **keras.Sequential** model and instantiate a stochastic gradient descent optimizer from **keras.optimizers**.
이제 **keras.Sequential** 로 네트워크를 빌드하고 **keras.optimizerrs** 로 SGD 옵티마이저를 생성합니다.

```python
network = Sequential([layers.Dense(256, activation='relu'),
                     layers.Dense(256, activation='relu'),
                     layers.Dense(256, activation='relu'),
                     layers.Dense(10)])
network.build(input_shape=(None, 28*28))
network.summary()

optimizer = optimizers.SGD(lr=0.01)
acc_meter = metrics.Accuracy()
```



Finally, we can iterate through our dataset and train our model.
In this example, we use **tf.GradientTape** to manually compute the gradients of the loss with respect to our network's trainable variables. GradientTape is just one of many ways to perform gradient steps in TensorFlow 2.0:
드디어, 데이터셋을 순회하고 모델을 학습 할 수 있게 되었습니다.
이 예제에서는 **tf.GradientTape**를 사용하여 네트워크의 학습 가능 변수에 대한 손실 그라디언트를 수동으로 계산합니다.

- **Tf.GradientTape:** Manually computes loss gradients with respect to given variables by recording operations within its context manager. This is the most flexible way to perform optimizer steps, as we can work directly with gradients and don't need a pre-defined Keras model or loss function.
- **Tf.GradientTape:** 컨텍스트 관리자 내에서 작업을 기록하여 주어진 변수에 대한 손실 그라디언트를 수동으로 계산합니다. 이는 그라디언트로 직접 작업 할 수 있고 미리 정의 된 Keras 모델 또는 손실 함수가 필요 없기 때문에 옵티 마이저 단계를 수행하는 가장 유연한 방법입니다.
- **Model.train():** Keras's built-in function for iterating through a dataset and fitting a Keras.Model on it. This is often the best choice for training a Keras model and comes with options for progress bar displays, validation splits, multiprocessing, and generator support.
- **Model.train():** 데이터 세트를 반복하고 Keras.Model을 피팅하는 Keras의 내장 함수. 이것은 종종 Keras 모델 교육에 가장 적합한 선택이며 진행률 표시 줄, 검증 분할, 다중 처리 및 생성기 지원에 대한 옵션이 제공됩니다.
- **Optimizer.minimize():** Computes and differentiates through a given loss function and performs a step to minimize it with gradient descent. This method is easy to implement, and can be conveniently slapped onto any existing computational graph to make a working optimization step.
- **Optimizer.minimize():** 주어진 손실 함수를 계산하고 차등화하고 기울기 강하로 최소화하는 단계를 수행합니다. 이 방법은 구현하기 쉽고 기존의 계산 그래프에 편리하게 적용 할 수 있으므로 작업 최적화 단계를 수행 할 수 있습니다.

```python
for step, (x,y) in enumerate(db):

    with tf.GradientTape() as tape:
        # [b, 28, 28] => [b, 784]
        x = tf.reshape(x, (-1, 28*28))
        # [b, 784] => [b, 10]
        out = network(x)
        # [b] => [b, 10]
        y_onehot = tf.one_hot(y, depth=10)
        # [b, 10]
        loss = tf.square(out-y_onehot)
        # [b]
        loss = tf.reduce_sum(loss) / 32


    acc_meter.update_state(tf.argmax(out, axis=1), y)

    grads = tape.gradient(loss, network.trainable_variables)
    optimizer.apply_gradients(zip(grads, network.trainable_variables))


    if step % 200==0:

        print(step, 'loss:', float(loss), 'acc:', acc_meter.result().numpy())
        acc_meter.reset_states()
```

# HowTO

Try it for yourself!

```
python main.py
``` 