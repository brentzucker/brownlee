import tensorflow as tf

# declare 2 floating point scalars
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

# simple symbolic expression
add = tf.add(a, b)

# bind values to a, b; then evaluate c
sess = tf.Session()
binding = {a: 1.5, b: 2.5}
c = sess.run(add, feed_dict=binding)
print(c)
