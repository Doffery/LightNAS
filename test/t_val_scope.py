import tensorflow as tf

def callCS(reuse):
    with tf.variable_scope("out", reuse=reuse):
        with tf.variable_scope("in1"):
            w1 = tf.get_variable("v1", [1])
            ww1 = tf.Variable([1], name='we1')
            c2d = tf.nn.conv2d(tf.random_normal([1,10,10,10]), 
                    tf.random_normal([2,10,10,10]), 
                    strides=[1, 2, 2, 1], padding='SAME')
        with tf.variable_scope("in1", reuse=tf.AUTO_REUSE):
            w2 = tf.get_variable("v2", [1])
            # w2 = tf.Variable([1], name='we2')
    print(w1.name)
    print(ww1.name)
    print(w2.name)
    print(c2d.name)

# with tf.variable_scope('a', reuse=tf.AUTO_REUSE):
callCS(False)
callCS(True)

def foo():
  with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
    v = tf.get_variable("v", [1])
  return v

v1 = foo()  # Creates v.
v2 = foo()  # Gets the same, existing v.
print(v1.name)
print(v2.name)
