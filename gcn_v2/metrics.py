import tensorflow as tf


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=tf.stop_gradient(labels))
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(input_tensor=mask)
    loss *= mask
    return tf.reduce_mean(input_tensor=loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(input=preds, axis=1), tf.argmax(input=labels, axis=1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(input_tensor=mask)
    accuracy_all *= mask
    return tf.reduce_mean(input_tensor=accuracy_all)
