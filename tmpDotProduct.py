import tensorflow as tf
numClass = 5
margin = 1
images = tf.constant([1.0, 2.0, 3,
                 4.0, 5.0, 6.0], shape=[2, 3], dtype=tf.float32)
classes = tf.constant([1.0, 2.0, 3.0,
                       4.0, 5.0, 6.0,
                       7.0, 8.0, 9.0,
                       10.0, 11.0, 12.0,
                       13.0, 14.0, 15.0], shape=[5, 3], dtype=tf.float32)
y = tf.constant([0, 1], shape=[2], dtype=tf.int32)

imageValue = tf.expand_dims(tf.reduce_sum(tf.square(images), axis=1), axis=1)
classValue = tf.expand_dims(tf.reduce_sum(tf.square(classes), axis=1), axis=0)
prodValue = tf.multiply(imageValue, classValue)

images = tf.expand_dims(images, axis=1)
out = tf.multiply(images, classes)
out1 = tf.reduce_sum(out, axis=2)
out1 = tf.divide(out1, prodValue)
correctClass = tf.tile(tf.reduce_max(tf.multiply(out1, tf.one_hot(y, numClass)), axis=1, keep_dims=True), [1, numClass])

cost = margin - correctClass + out1
cost2 = tf.maximum(cost, 0.0)
cost2 = tf.multiply(cost2, tf.one_hot(y, numClass, on_value=0.0, off_value=1.0))
cost3 = tf.reduce_sum(cost2, axis=1)


sess = tf.Session()
print(sess.run(imageValue))
print(sess.run(classValue))
print(sess.run(prodValue))


print(sess.run(out))
print(sess.run(out1))
print(sess.run(correctClass))
print(sess.run(cost))
print(sess.run(cost2))
print(sess.run(cost3))
