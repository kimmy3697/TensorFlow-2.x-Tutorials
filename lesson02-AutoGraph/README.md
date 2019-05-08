# AutoGraph


Compare static graph using @tf.function VS dynamic graph.
@tf.function 과 dynamic graph 사용시 static graph 비교

AutoGraph helps you write complicated graph code using normal Python. Behind the scenes, AutoGraph automatically transforms your code into the equivalent TensorFlow graph code.
AutoGraph는 일반 파이썬을 통해 복잡한 그래프 코드를 작성할 때 도움이 됩니다. 
오토 그래프는 자동드로 작성한 코드를 적합한 텐서플로우 그래프 코드로 변환합니다.

Let's take a look at TensorFlow graphs and how they work.
텐서플로우 그래프를 살펴보고 어떻게 동작하는지 알아 봅시다.

```python
ReLU_Layer = tf.keras.layers.Dense(100, input_shape=(784,), activation=tf.nn.relu)
Logit_Layer = tf.keras.layers.Dense(10, input_shape=(100,))

# X and y are labels and inputs
```

<img src="graph.gif" align="left" width="302" height="538">

**TensorFlow 1.0:** Operations are added as nodes to the computational graph and are not actually executed until we call session.run(), much like defining a function that doesn't run until it is called.
**텐서플로우1.0:** Operations 는 computational 그래프에 노드로 연결되고 session.run()을 실행
하기 전까지 실제로 실행되지 않습니다, 마치 정의한 함수가 호출전에 실행되지 않는것과 같습니다.
```python
SGD_Trainer = tf.train.GradientDescentOptimizer(1e-2)

inputs = tf.placeholder(tf.float32, shape=[None, 784])
labels = tf.placeholder(tf.int16, shape=[None, 10])
hidden = ReLU_Layer(inputs)
logits = Logit_Layer(hidden)
entropy = tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=labels)
loss = tf.reduce_mean(entropy)
train_step = SGD_Trainer.minimize(loss, 
    var_list=ReLU_Layer.weights+Logit_Layer.weights)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for step in range(1000):
    sess.run(train_step, feed_dict={inputs:X, labels:y})
```

**TensorFlow 2.0:** Operations are executed directly and the computational graph is built on-the-fly. However, we can still write functions and pre-compile computational graphs from them like in TF 1.0 using the *@tf.function* decorator, allowing for faster execution.
**텐서플로우 2.0:** Operations 는 직접 실행되고 computational 그래프가 즉석에서 빌드 됩니다.
또한, TF 1.0 에서 처럼 @tf.function 데코레이터를 통해 함수를 작성하고 computational graph를 pre-compile 할 수 있습니다.

```python
SGD_Trainer = tf.optimizers.SGD(1e-2)

@tf.function
def loss_fn(inputs=X, labels=y):
    hidden = ReLU_Layer(inputs)
    logits = Logit_Layer(hidden)
    entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=labels)
    return tf.reduce_mean(entropy)

for step in range(1000):
    SGD_Trainer.minimize(loss_fn, 
        var_list=ReLU_Layer.weights+Logit_Layer.weights)
```

# HowTO

```
python main.py
```

and you will see some computation cost between static graph and dynamic graph.